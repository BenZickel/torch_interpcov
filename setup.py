import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name='torch_interpcov',
    version='0.0.1',
    long_description=long_description,
    long_description_content_type="text/markdown",
    description='Covariance Matrix Interpolation in PyTorch',
    python_requires='>=3.6',
    packages=setuptools.find_packages(),
    author='Ben Zickel',
    license='BSD')
