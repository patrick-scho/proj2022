import os
import mitsuba
mitsuba.set_variant('scalar_rgb')
from mitsuba.core import Thread
from mitsuba.core.xml import load_file

# Absolute or relative path to the XML file
filename = 'head.xml'

# Add the scene directory to the FileResolver's search path
Thread.thread().file_resolver().append(os.path.dirname(filename))

# Load the scene for an XML file
scene = load_file(filename)

# Get the scene's sensor (if many, can pick one by specifying the index)
sensor = scene.sensors()[0]

# Call the scene's integrator to render the loaded scene with the desired sensor
scene.integrator().render(scene, sensor)

# The rendered data is stored in the film
film = sensor.film()

# Write out data as high dynamic range OpenEXR file
film.set_destination_file('hello.exr')
film.develop()

# Write out a tone-mapped JPG of the same rendering
from mitsuba.core import Bitmap, Struct
img = film.bitmap(raw=True).convert(Bitmap.PixelFormat.RGB, Struct.Type.UInt8, srgb_gamma=True)
img.write('hello.jpg')