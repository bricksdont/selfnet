from setuptools import setup

setup(
   name='selfnet',
   version='1.0',
   description='Feed-forward neural networks written in numpy',
   author='Mathias Müller',
   author_email='mmueller@cl.uzh.ch',
   packages=['selfnet'],
   license="lgpl",
   install_requires=['numpy'],
)
