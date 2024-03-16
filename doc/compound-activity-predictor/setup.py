from setuptools import setup, find_packages
import os
import sys
import traceback

try:
    # Get the long description from the README file
    with open(os.path.join(os.path.dirname(__file__), 'README.md'), encoding='utf-8') as f:
        long_description = f.read()
except Exception as e:
    print(f"Error reading README.md: {e}")
    traceback.print_exc()
    long_description = "Compound Activity Predictor"

try:
    # Get the required packages from the requirements file
    with open(os.path.join(os.path.dirname(__file__), 'requirements.txt'), encoding='utf-8') as f:
        requirements = f.read().splitlines()
except Exception as e:
    print(f"Error reading requirements.txt: {e}")
    traceback.print_exc()
    requirements = [
        'dgl',
        'dgllife',
        'rdkit-pypi',
        'chembl-webresource-client',
        'scikit-learn',
        'numpy',
        'pandas',
        'matplotlib',
        'pyscf',
        'optuna',
    ]

def main():
    """
    Main function to set up the compound activity predictor package.
    """
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
