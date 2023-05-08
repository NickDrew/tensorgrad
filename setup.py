import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="tensorgrad",
    version="0.0.1",
    author="Nicholas Drew",
    author_email="nicholas.drew@hotmail.com",
    description="A tiny tensor-valued autograd engine with a small PyTorch-like neural network library on top, building on work done by Andrej Karpathy",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/nickdrew/tensorgrad",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
)
