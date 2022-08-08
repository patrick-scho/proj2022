from __future__ import print_function
#%matplotlib inline
import argparse
import os
import random
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.nn.functional
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.utils as vutils
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

import deodr
from deodr import differentiable_renderer
from deodr.pytorch import triangulated_mesh_pytorch
from deodr.pytorch import differentiable_renderer_pytorch
from scipy.spatial.transform import Rotation
import trimesh
import imageio.v3 as iio

import dnnlib
import legacy
import PIL.Image



def ColoredTriMeshPytorch_from_trimesh(mesh):
    """Get the vertex colors, texture coordinates, and material properties
    from a :class:`~trimesh.base.Trimesh`.
    """
    colors = None
    uv = None
    texture = None

    # If the trimesh visual is undefined, return none for both

    # Process vertex colors
    if mesh.visual.kind == "vertex":
        colors = mesh.visual.vertex_colors.copy()
        if colors.ndim == 2 and colors.shape[1] == 4:
            colors = colors[:, :3]
        colors = colors.astype(np.float) / 255

    # Process face colors
    elif mesh.visual.kind == "face":
        raise BaseException(
            "not supported yet, will need antialisaing at the seams"
        )

    # Process texture colors
    elif mesh.visual.kind == "texture":
        # Configure UV coordinates
        if mesh.visual.uv is not None:

            texture = np.array(mesh.visual.material.image) / 255
            texture.setflags(write=0)

            if texture.shape[2] == 4:
                texture = texture[:, :, :3]  # removing alpha channel

            uv = (
                np.column_stack(
                    (
                        (mesh.visual.uv[:, 0]) * texture.shape[1],
                        (1 - mesh.visual.uv[:, 1]) * texture.shape[0],
                    )
                )
                - 0.5
            )

    # merge identical 3D vertices even if their uv are different to keep surface
    # manifold. Trimesh seems to split vertices that have different uvs (using
    # unmerge_faces texture.py), making the surface not watertight, while there
    # were only seems in the texture.

    vertices, return_index, inv_ids = np.unique(
        mesh.vertices, axis=0, return_index=True, return_inverse=True
    )
    faces = inv_ids[mesh.faces].astype(np.int32)
    if colors is not None:
        colors2 = colors[return_index, :]
        if np.any(colors != colors2[inv_ids, :]):
            raise (
                BaseException(
                    "vertices at the same 3D location should have the same color"
                    "for the rendering to be differentiable"
                )
            )
    else:
        colors2 = None

    return triangulated_mesh_pytorch.ColoredTriMeshPytorch(
        faces,
        torch.tensor(vertices),
        clockwise=False,
        faces_uv=torch.tensor(mesh.faces),
        uv=uv,
        texture=texture,
        colors=colors2,
    )

class PerspectiveCameraPytorch(differentiable_renderer_pytorch.CameraPytorch):
    """Camera with perspective projection."""

    def __init__(self, width, height, fov, camera_center, rot=None, distortion=None):
        """Perspective camera constructor.

        - width: width of the camera in pixels
        - height: eight of the camera in pixels
        - fov: horizontal field of view in degrees
        - camera_center: center of the camera in world coordinate system
        - rot: 3x3 rotation matrix word to camera (x_cam = rot.dot(x_world))\
            default to identity
        - distortion: distortion parameters
        """
        if rot is None:
            rot = np.eye(3)
        focal = 0.5 * width / np.tan(0.5 * fov * np.pi / 180)
        focal_x = focal
        pixel_aspect_ratio = 1
        focal_y = focal * pixel_aspect_ratio
        trans = -rot.T.dot(camera_center)
        cx = width / 2
        cy = height / 2
        intrinsic = np.array([[focal_x, 0, cx], [0, focal_y, cy], [0, 0, 1]])
        extrinsic = np.column_stack((rot, trans))
        super().__init__(
            extrinsic=torch.tensor(extrinsic),
            intrinsic=torch.tensor(intrinsic),
            distortion=distortion,
            width=width,
            height=height,
        )

def default_scene(
    obj_file, width, height, use_distortion=True, integer_pixel_centers=True
):
    mesh_trimesh = trimesh.load(obj_file)
    mesh = ColoredTriMeshPytorch_from_trimesh(mesh_trimesh)
    # rot = Rotation.from_euler("xyz", [180, 0, 0], degrees=True).as_matrix()
    rot = Rotation.from_euler("xyz", [180, 0, 0], degrees=True).as_matrix()

    camera = PerspectiveCameraPytorch(
        width, height, 60, (0, 1.7, 0.3), rot)
    if use_distortion:
        camera.distortion = np.array([-0.5, 0.5, 0, 0, 0])

    bg_color = np.array((0.8, 0.8, 0.8))
    scene = differentiable_renderer_pytorch.Scene3DPytorch()
    scene.integer_pixel_centers = integer_pixel_centers
    light_ambient = 0
    light_directional = 0.4 * np.array([1, -3, -0.5])
    scene.set_light(light_directional=light_directional, light_ambient=light_ambient)
    scene.set_mesh(mesh)
    scene.set_background_color(bg_color)
    return scene, camera



def render_rgb(texture):
    scene.mesh.texture = texture
    image = scene.render(camera)
    return image

def tensor2texture(texture):
    return np.moveaxis(texture, 0, -1)
def texture2tensor(tensor):
    return np.moveaxis(tensor, -1, 0)


def render_batch(batch):
    cpu_batch = batch.detach().cpu()
    cpu_batch_rendered = []
    for i in range(len(cpu_batch)):
        texture = tensor2texture(np.array(cpu_batch[i])/2+0.5)

        rendered = render_rgb(texture)

        ax1.imshow(texture)
        ax2.imshow(rendered)
        plt.pause(0.001)

        cpu_batch_rendered.append(
            texture2tensor(
                rendered
            )
        )
    return torch.FloatTensor(cpu_batch_rendered)



# custom weights initialization called on netG and netD
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)

        


class Generator(nn.Module):
    def __init__(self, ngpu):
        super(Generator, self).__init__()
        self.ngpu = ngpu
        self.main = nn.Sequential(
            # input is Z, going into a convolution
            nn.ConvTranspose2d( nz, ngf * 8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(ngf * 8),
            nn.ReLU(True),
            # state size. (ngf*8) x 4 x 4
            nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 4),
            nn.ReLU(True),
            # state size. (ngf*4) x 8 x 8
            nn.ConvTranspose2d( ngf * 4, ngf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 2),
            nn.ReLU(True),
            # state size. (ngf*2) x 16 x 16
            nn.ConvTranspose2d( ngf * 2, ngf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf),
            nn.ReLU(True),
            # state size. (ngf) x 32 x 32
            nn.ConvTranspose2d( ngf, nc, 4, 2, 1, bias=False),
            nn.Tanh()
            # state size. (nc) x 64 x 64
        )

    def forward(self, input):
        return self.main(input)



class Discriminator(nn.Module):
    def __init__(self, ngpu):
        super(Discriminator, self).__init__()
        self.ngpu = ngpu
        self.main = nn.Sequential(
            # input is (nc) x 64 x 64
            nn.Conv2d(nc, ndf, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf) x 32 x 32
            nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*2) x 16 x 16
            nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*4) x 8 x 8
            nn.Conv2d(ndf * 4, ndf * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 8),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*8) x 4 x 4
            nn.Conv2d(ndf * 8, 1, 4, 1, 0, bias=False),
            nn.Sigmoid()
        )

    def forward(self, input):
        return self.main(input)





if __name__ == '__main__':
    # Set random seed for reproducibility
    manualSeed = 999
    #manualSeed = random.randint(1, 10000) # use if you want new results
    print("Random Seed: ", manualSeed)
    random.seed(manualSeed)
    torch.manual_seed(manualSeed)


    # Root directory for dataset
    dataroot = "celeba"

    # Number of workers for dataloader
    workers = 2

    # Batch size during training
    batch_size = 128

    # Spatial size of training images. All images will be resized to this
    #   size using a transformer.
    image_size = 64

    # Number of channels in the training images. For color images this is 3
    nc = 3

    # Size of z latent vector (i.e. size of generator input)
    nz = 100

    # Size of feature maps in generator
    ngf = 64

    # Size of feature maps in discriminator
    ndf = 64

    # Number of training epochs
    num_epochs = 5

    # Learning rate for optimizers
    lr = 0.0002

    # Beta1 hyperparam for Adam optimizers
    beta1 = 0.5

    # Number of GPUs available. Use 0 for CPU mode.
    ngpu = 1


    # We can use an image folder dataset the way we have it setup.
    # Create the dataset
    dataset = dset.ImageFolder(root=dataroot,
                            transform=transforms.Compose([
                                transforms.Resize(image_size),
                                transforms.CenterCrop(image_size),
                                transforms.ToTensor(),
                                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                            ]))
    # Create the dataloader
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size,
                                            shuffle=True, num_workers=workers)

    # Decide which device we want to run on
    device = torch.device("cuda:0" if (torch.cuda.is_available() and ngpu > 0) else "cpu")



    # Create the generator
    netG = Generator(ngpu).to(device)

    # Handle multi-gpu if desired
    if (device.type == 'cuda') and (ngpu > 1):
        netG = nn.DataParallel(netG, list(range(ngpu)))

    # Apply the weights_init function to randomly initialize all weights
    #  to mean=0, stdev=0.02.
    netG.apply(weights_init)

    # Print the model
    print(netG)




    # Create the Discriminator
    netD = Discriminator(ngpu).to(device)

    # Handle multi-gpu if desired
    if (device.type == 'cuda') and (ngpu > 1):
        netD = nn.DataParallel(netD) #, list(range(ngpu)))

    # Apply the weights_init function to randomly initialize all weights
    #  to mean=0, stdev=0.2.
    netD.apply(weights_init)

    # Print the model
    print(netD)


    
    obj_filename = "filled_not_boerner.obj"
    scene, camera = default_scene(obj_filename, width=64, height=64)
    fig, (ax1, ax2) = plt.subplots(1, 2)
    plt.show(block = False)




    # Initialize BCELoss function
    criterion = nn.BCELoss()

    # Create batch of latent vectors that we will use to visualize
    #  the progression of the generator
    fixed_noise = torch.randn(64, nz, 1, 1, device=device)

    # Establish convention for real and fake labels during training
    real_label = 1.
    fake_label = 0.

    # Setup Adam optimizers for both G and D
    optimizerD = optim.Adam(netD.parameters(), lr=lr, betas=(beta1, 0.999))
    optimizerG = optim.Adam(netG.parameters(), lr=lr, betas=(beta1, 0.999))






    # Training Loop

    # Lists to keep track of progress
    img_list = []
    G_losses = []
    D_losses = []
    iters = 0

    print("Starting Training Loop...")
    # For each epoch
    for epoch in range(num_epochs):
        # For each batch in the dataloader
        for i, data in enumerate(dataloader, 0):

            ############################
            # (1) Update D network: maximize log(D(x)) + log(1 - D(G(z)))
            ###########################
            ## Train with all-real batch
            netD.zero_grad()
            # Format batch
            real_cpu = data[0].to(device)
            b_size = real_cpu.size(0)
            label = torch.full((b_size,), real_label, dtype=torch.float, device=device)
            # Forward pass real batch through D
            output = netD(real_cpu).view(-1)
            # Calculate loss on all-real batch
            errD_real = criterion(output, label)
            # Calculate gradients for D in backward pass
            errD_real.backward()
            D_x = output.mean().item()

            ## Train with all-fake batch
            # Generate batch of latent vectors
            noise = torch.randn(b_size, nz, 1, 1, device=device)
            # Generate fake image batch with G
            fake = netG(noise)
            fake = render_batch(fake).to(device)
            label.fill_(fake_label)
            # Classify all fake batch with D
            output = netD(fake.detach()).view(-1)
            # Calculate D's loss on the all-fake batch
            errD_fake = criterion(output, label)
            # Calculate the gradients for this batch, accumulated (summed) with previous gradients
            errD_fake.backward()
            D_G_z1 = output.mean().item()
            # Compute error of D as sum over the fake and the real batches
            errD = errD_real + errD_fake
            # Update D
            optimizerD.step()

            ############################
            # (2) Update G network: maximize log(D(G(z)))
            ###########################
            netG.zero_grad()
            label.fill_(real_label)  # fake labels are real for generator cost
            # Since we just updated D, perform another forward pass of all-fake batch through D
            output = netD(fake).view(-1)
            # Calculate G's loss based on this output
            errG = criterion(output, label)
            # Calculate gradients for G
            errG.backward()
            D_G_z2 = output.mean().item()
            # Update G
            optimizerG.step()

            # Output training stats
            if i % 50 == 0:
                print('[%d/%d][%d/%d]\tLoss_D: %.4f\tLoss_G: %.4f\tD(x): %.4f\tD(G(z)): %.4f / %.4f'
                    % (epoch, num_epochs, i, len(dataloader),
                        errD.item(), errG.item(), D_x, D_G_z1, D_G_z2))

            # Save Losses for plotting later
            G_losses.append(errG.item())
            D_losses.append(errD.item())

            # Check how the generator is doing by saving G's output on fixed_noise
            if (iters % 500 == 0) or ((epoch == num_epochs-1) and (i == len(dataloader)-1)):
                with torch.no_grad():
                    fake = netG(fixed_noise).detach().cpu()
                img_list.append(vutils.make_grid(fake, padding=2, normalize=True))
                img_list.append(vutils.make_grid(fake, padding=2, normalize=True))

            iters += 1





    torch.save(netG.state_dict(), "gen.pt")



    plt.figure(figsize=(10,5))
    plt.title("Generator and Discriminator Loss During Training")
    plt.plot(G_losses,label="G")
    plt.plot(D_losses,label="D")
    plt.xlabel("iterations")
    plt.ylabel("Loss")
    plt.legend()
    plt.show()

    #%%capture
    fig = plt.figure(figsize=(8,8))
    plt.axis("off")
    ims = [[plt.imshow(np.transpose(i,(1,2,0)), animated=True)] for i in img_list]
    ani = animation.ArtistAnimation(fig, ims, interval=1000, repeat_delay=1000, blit=True)

    #HTML(ani.to_jshtml())

    # Grab a batch of real images from the dataloader
    real_batch = next(iter(dataloader))

    # Plot the real images
    plt.figure(figsize=(15,15))
    plt.subplot(1,2,1)
    plt.axis("off")
    plt.title("Real Images")
    plt.imshow(np.transpose(vutils.make_grid(real_batch[0].to(device)[:64], padding=5, normalize=True).cpu(),(1,2,0)))

    # Plot the fake images from the last epoch
    plt.subplot(1,2,2)
    plt.axis("off")
    plt.title("Fake Images")
    plt.imshow(np.transpose(img_list[-1],(1,2,0)))
    plt.show()