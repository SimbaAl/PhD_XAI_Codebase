from setuptools import setup, find_packages

# Read the README for the long description
with open('README.md', 'r', encoding='utf-8') as f:
    long_description = f.read()

setup(
    name="grace",
    version="0.1.0",
    author="Simbarashe Aldrin Ngorima",
    author_email="aldringorima@gmail.com",
    description="Layer Relevance Propagation-based XAI Scheme for Channel Estimation",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/username/grace-channel-estimation",
    project_urls={
        "Bug Tracker": "https://github.com/username/grace-channel-estimation/issues",
    },
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Information Analysis",
    ],
    packages=find_packages(include=['src', 'src.*']),
    python_requires=">=3.11",
    install_requires=[
        'torch>=2.0.0',
        'numpy>=1.24.0',
        'scipy>=1.10.0',
        'h5py>=3.8.0',
        'matplotlib>=3.7.0',
        'seaborn>=0.12.0',
        'scikit-learn>=1.2.0',
        'pandas>=2.0.0',
        'mat73>=0.60'
    ],
    extras_require={
        'dev': [
            'pytest>=7.3.1',
            'pytest-cov>=4.1.0',
            'black>=23.3.0',
            'flake8>=6.0.0',
            'mypy>=1.3.0',
            'sphinx>=7.0.0',
            'sphinx-rtd-theme>=1.2.0'
        ]
    },
    entry_points={
        'console_scripts': [
            'grace-train=experiments.train_model:main',
            'grace-relevance=experiments.compute_relevance:main',
            'grace-evaluate=experiments.evaluate_features:main',
        ],
    },
    include_package_data=True,
    license="MIT",
    platforms="any",
    keywords=[
        "deep-learning",
        "channel-estimation",
        "explainable-ai",
        "layer-relevance-propagation",
        "neural-networks",
        "vehicular-communications"
    ],
)