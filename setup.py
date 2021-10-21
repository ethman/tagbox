from setuptools import setup, find_packages

with open('README.md') as f:
    long_description = f.read()

with open('requirements.txt') as f:
    requirements = f.read().splitlines()

setup(
    name='tagbox',
    version="0.0.1",
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Environment :: Console',
        'Intended Audience :: Developers',
        'Intended Audience :: Education',
        'Intended Audience :: Science/Research',
        'Natural Language :: English',
        'Operating System :: OS Independent',
        'Programming Language :: Python :: 3.7',
        'Topic :: Artistic Software',
        'Topic :: Multimedia',
        'Topic :: Multimedia :: Sound/Audio',
        'Topic :: Multimedia :: Sound/Audio :: Editors',
        'Topic :: Software Development :: Libraries',
    ],
    description='TagBox: VQGAN+CLIP for music!',
    long_description=long_description,
    long_description_content_type='text/markdown',
    author='Ethan Manilow',
    author_email='',
    url='https://github.com/ethman/sota-music-tagging-models',
    license='MIT',
    packages=find_packages(),
    keywords=['jukebox', 'music tagging', 'VQGAN+CLIP', 'music', 'machine learning'],
    install_requires=requirements,
)