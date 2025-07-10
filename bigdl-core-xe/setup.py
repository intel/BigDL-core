import os
import setuptools

VERSION = open(os.path.join(os.path.dirname(os.path.realpath(__file__)),
               'version.txt'), 'r').read().strip()

setuptools.setup(
    name='bigdl-core',
    version=VERSION,
    package_dir={".": "."},
    packages=["."],
    package_data={".": ["*.pyd", "*.so"]},
    include_package_data=True,
    ext_modules=[setuptools.Extension(name='no_ext', sources=[])]
)