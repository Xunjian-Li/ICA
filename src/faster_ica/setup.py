#! /usr/bin/env python
from setuptools import find_packages
from numpy.distutils.core import setup

descr = """Maximum likelihood ICA algorithms"""

DISTNAME = 'ml_ica'
DESCRIPTION = descr
MAINTAINER = 'Pierre Ablin'
MAINTAINER_EMAIL = 'pierre.ablin@inria.fr'
LICENSE = 'BSD (3-clause)'
DOWNLOAD_URL = 'https://github.com/pierreablin/faster-ica.git'
VERSION = '0.1.dev0'

if __name__ == "__main__":
    setup(name=DISTNAME,
          maintainer=MAINTAINER,
          maintainer_email=MAINTAINER_EMAIL,
          description=DESCRIPTION,
          license=LICENSE,
          version=VERSION,
          download_url=DOWNLOAD_URL,
          long_description=open('README.rst').read(),
          classifiers=[
              'Intended Audience :: Science/Research',
              'Intended Audience :: Developers',
              'License :: OSI Approved',
              'Programming Language :: Python',
              'Topic :: Software Development',
              'Topic :: Scientific/Engineering',
              'Operating System :: Microsoft :: Windows',
              'Operating System :: POSIX',
              'Operating System :: Unix',
              'Operating System :: MacOS',
          ],
          platforms='any',
          packages=find_packages(),
          )
