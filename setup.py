import shutil
import os

from distutils.core import setup

with open('requirements.txt') as f:
    REQUIREMENTS = [
        line.strip() for line in f.readlines() if line.strip()]


setup(
    name='yvolv',
    version='0.1',
    author='Jeff Schecter',
    author_email='jeffrey.schecter@gmail.com',
    license='MIT',
    description='Alife simulator',
    url='https://github.com/jeffschecter/yvolv',
    keywords=['yvolv', 'alife'],
    packages=["yvolv"],
    classifiers=[],
    install_requires=REQUIREMENTS
)