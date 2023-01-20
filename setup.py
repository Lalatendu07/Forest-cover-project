from setuptools import find_packages,setup
from typing import List

REQUIREMENT_FILE = 'requirements.txt'
HYPHEN_E_DOT = '-e .'

def get_requirements()->List[str]:
    with open(REQUIREMENT_FILE) as requirement_file :
        requirement_list = requirement_file.readlines()
    requirement_list = [requirement_name.replace('\n','') for requirement_name in requirement_list]
        
    if HYPHEN_E_DOT in requirement_list:
        requirement_list.remove(HYPHEN_E_DOT)
    return requirement_list


setup(
    name = 'forest_cover_type',
    version = '0.0.1',
    author='Lalatendu dalai',
    author_email='lalatendu.4391@gmail.com',
    packages=find_packages(),
    install_requires = get_requirements()

)