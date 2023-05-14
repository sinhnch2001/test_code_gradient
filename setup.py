from distutils.core import setup
from setuptools import find_packages

NAME = 'dialogue_state.baseline.v1'
VERSION = '0.1.1'
LICENSE = 'MIT'
DESCRIPTION = 'State Prediction for Task-oriented task'

with open('README.md', encoding="utf-8") as file:
    description = file.read()

setup(
    name=NAME,
    version=VERSION,
    packages=find_packages('src'),
    license=LICENSE,
    zip_safe=True,
    description=DESCRIPTION,
    long_description=description,
    long_description_content_type='text/markdown',
    author='Tien V. Nguyen',
    author_email='tiennv@gradients.host',
    url='https://github.com/gradients-tech/dialogstate',
    python_require='>=3.8.15',
    keywords=[],
    install_requires=[
        'accelerate~=0.17.0',
        'datasets>=2.8.0',
        'filelock==3.10.3',
        'nltk>=3.7',
        'numpy>=1.17.5',
        'pandas>=1.3.2',
        'setuptools>=62.0.0',
        '--extra-index-url https://download.pytorch.org/whl/cu113',
        'torch==1.12.0+cu113',
        'tqdm~=4.64.1',
        'transformers>=4.27.0',
        'typing_extensions>=4.1.0',
        'evaluate>=0.3.0',
        'pyarrow>=7.0.0',
        'rouge_score>=0.1'
    ],
    classifiers=[
        'Development Status :: 1 - Planning',
        'Intended Audience :: Developers',
        'Topic :: Software Development :: Build Tools',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9'
    ],
)