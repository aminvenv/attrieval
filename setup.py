from setuptools import setup, find_packages

setup(
    name='attrieval',
    version='0.1.0',
    description='A Python Library for the Evaluation of Attribution',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    author='amin',
    author_email='',
    url='',
    packages=find_packages(),
    install_requires=[
        # list your dependencies here
    ],
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.9',
)