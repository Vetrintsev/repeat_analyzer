from setuptools import setup, find_packages

setup(
    name='repeat_analyzer',
    version='0.1',
    description='Search for patterns of customer behavior by repeated incidents',
    long_description='Search for patterns of customer behavior by repeated incidents',
    classifiers=[
    'Development Status :: 3 - Alpha',
    'License :: OSI Approved :: MIT License',
    'Programming Language :: Python :: 3.6',
    'Topic :: Office/Business',
    ],
    keywords='incidents service repeats fcr csi',
    url='http://github.com/...',
    author='Anatoliy Vetrintsev',
    author_email='vetrintsev@gmail.com',
    license='MIT',
    packages=find_packages(),
    install_requires=[
      'pandas', 'numpy', 'tensorflow', 'keras', 'umap-learn', 'hdbscan', 'matplotlib'
    ],
    include_package_data=True,
    zip_safe=False
)