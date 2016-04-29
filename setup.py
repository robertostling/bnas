import sys
import os

from setuptools import setup

version = '0.1.dev1'
# TODO: use `git rev-parse --short HEAD` ?

long_description = '''
Software package to provide a friendlier interface to the Theano library for
neural networks, with a focus on Natural Language Processing applications.
'''

setup(
        name='bnas',
        version=version,
        description='Basic Neural Architecture Subprograms',
        long_description=long_description,
        url='https://github.com/robertostling/bnas',
        author='Robert Östling',
        license='GNU GPLv3',
        classifiers=[
            'Development Status :: 2 - Pre-Alpha',
            'Intended Audience :: Science/Research',
            'Topic :: Scientific/Engineering :: Artificial Intelligence',
            'Topic :: Text Processing :: Linguistic',
            'Programming Language :: Python :: 3'],
        keywords='neural networks theano',
        install_requires=['numpy', 'theano'])

