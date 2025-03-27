from setuptools import find_packages, setup
import sysconfig

package_name = 'cameras'

# Get the Python installation path
python_version = sysconfig.get_python_version()

setup(
    name=package_name,
    version='0.0.1',
    packages=find_packages(exclude=['test']),
    data_files=[
        # Use dynamic Python version and site-packages path
        (f'lib/python{python_version}/site-packages/{package_name}', ['cameras/left.ini', 'cameras/right.ini']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='ryandash',
    maintainer_email='ryandash@cmail.carleton.ca',
    description='ROS 2 package for camera feed with calibration info',
    license='MIT',
    entry_points={
        'console_scripts': [
            'camera_feeds_with_info = cameras.camera_feeds_with_info:main',
        ],
    },
)
