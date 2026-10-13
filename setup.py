
from setuptools import find_packages, setup  # Packages find karne aur project setup karne ke liye
from typing import List  # List data type ke type hints ke liye

HYPEN_E_DOT = "-e ."  # Editable installation ke liye requirement string


def get_requirements(file_path: str) -> List[str]:
    """
    Ye function requirements.txt file se packages ki list return karega.
    """

    requirements = []  # Empty list create kar rahe hain

    with open(file_path, "r") as file_obj:  # Requirements file ko read mode mein open karna
        requirements = file_obj.readlines()  # File ki saari lines read karna

        requirements = [
            req.replace("\n", "").strip()  # Har line se newline aur extra spaces remove karna
            for req in requirements
        ]

        if HYPEN_E_DOT in requirements:  # Check karna ki -e . list mein hai ya nahi
            requirements.remove(HYPEN_E_DOT)  # -e . ko list se remove karna

    return requirements  # Packages ki list return karna


setup(
    name="punammlproject",  # Project ka naam
    version="0.0.1",  # Project ka version
    author="Punam Vardhaiyya",  # Author ka naam
    author_email="punamvardhaiyya@gmail.com",  # Author ka email
    packages=find_packages(),  # Project ke Python packages automatically find karna
    install_requires=get_requirements("requirements.txt")  # Dependencies install karna
)
   