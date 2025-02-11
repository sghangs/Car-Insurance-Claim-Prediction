from setuptools import find_packages,setup
from typing import List 

def get_requirements() -> List[str]:
    """
    This function will return the list of all dependancies from requirements.txt file
    """
    requirement_list=[]

    try:
        with open("requirements.txt","r") as file:
            lines=file.readlines()
            for line in lines:
                requirement=line.strip()
                #Ignore the empty line and .e
                if requirement and requirement != "-e .":
                    requirement_list.append(requirement) 
    except FileNotFoundError:
        print("requirements.txt file is not found")

    return requirement_list

setup(
    name="CarInsuranceClaimPrediction",
    version="0.0.1",
    author="Sunny Ghangas",
    author_email="sunnyghangas098@gmail.com",
    packages=find_packages(),
    install_requires=get_requirements()
)