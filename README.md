from setuptools import setup, find_packages
import os
import sys
import traceback

def read_readme():
    """
    Read and return the contents of the README.md file.

    Returns:
        str: The contents of the README.md file.

    Raises:
        FileNotFoundError: If the README.md file is not found.
        Exception: If any other exception occurs while reading the file.
    """
    readme_path = os.path.join(os.path.dirname(__file__), 'README.md')
    try:
        with open(readme_path, encoding='utf-8') as f:
            long_description = f.read()
    except FileNotFoundError as e:
        print(f"Error: README.md file not found at {readme_path}.")
        traceback.print_exc()
        long_description = "Compound Activity Predictor"
    except Exception as e:
        print(f"Error reading README.md: {e}")
        traceback.print_exc()
        long_description = "Compound Activity Predictor"

    return long_description

def read_requirements(file_path):
    """
    Read and return the required packages from the given file.

    Args:
        file_path (str): The path to the requirements file.

    Returns:
        list: A list of required packages.

    Raises:
        FileNotFoundError: If the requirements file is not found.
        Exception: If any other exception occurs while reading the file.
    """
    try:
        with open(file_path, encoding='utf-8') as f:
            requirements = f.read().splitlines()
    except FileNotFoundError as e:
        print(f"Error: {file_path} not found.")
        traceback.print_exc()
        requirements = []
    except Exception as e:
        print(f"Error reading {file_path}: {e}")
        traceback.print_exc()
        requirements = []

    return requirements

def main():
    """
    Main function to set up the compound activity predictor package.
    """
    long_description = read_readme()
    requirements_file = os.path.join(os.path.dirname(__file__), 'requirements.txt')
    requirements = read_requirements(requirements_file)

    setup(
        name='compound-activity-predictor',
        version='0.1',
        description='Compound Activity Predictor',
        long_description=long_description,
        long_description_content_type='text/markdown',
        author='Your Name',
        author_email='your.email@example.com',
        url='https://github.com/your-username/compound-activity-predictor',
        packages=find_packages(),
        install_requires=requirements,
        entry_points={
            'console_scripts': [
                'compound-activity-predictor=main:main',
            ],
        },
    )

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"Error setting up the package: {e}")
        traceback.print_exc()
        sys.exit(1)
