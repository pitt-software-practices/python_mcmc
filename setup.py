from setuptools import setup, find_packages

setup(
    name='python_mcmc',
    version='0.1.0',
    packages=find_packages(include=['python_mcmc', 'python_mcmc.*'])
    install_requires=['numpy', 
        'matplotlib',
        'lmfit',
        'scipy',
        'corner',
        'tqdm'
    )
