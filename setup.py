from setuptools import setup

setup(
    name='tfword2vec',
    version='0.1',
    description='A basic Word2Vec implementation on tensorflow',
    url='http://github.com/oktopac/tfword2vec',
    author='Oktopac',
    license='Apache2.0',
    packages=['tfword2vec'],
    test_suite='nose.collector',
    tests_require=['nose'],
    install_requires=[
        'numpy',
        'tensorflow'
      ]
    )