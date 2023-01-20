from setuptools import find_packages,setup

def get_requirements():...

setup(
    name = 'forest_cover_type',
    version = '0.0.1',
    author='Lalatendu dalai',
    author_email='lalatendu.4391@gmail.com',
    packages=find_packages(),
    install_requires = get_requirements()

)