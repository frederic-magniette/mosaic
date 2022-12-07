import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name = 'ml-mosaic',
    version = '0.0.4',
    author = 'Frederic Magniette',
    author_email = 'magniette@llr.in2p3.fr',
    scripts = ['mosaic/bin/mosaic_run',
               'mosaic/bin/mosaic_rerun',
               'mosaic/bin/mosaic_savedb',
               'mosaic/bin/mosaic_pause',
               'mosaic/bin/mosaic_resume',
               'mosaic/bin/mosaic_status',
               'mosaic/bin/mosaic_generate',
               'mosaic/bin/mosaic_plotloss',
               'mosaic/bin/mosaic_metaplot',
               'mosaic/bin/mosaic_perfanalysis',
               'mosaic/bin/mosaic_perfanalysis_by_config',
               'mosaic/lib/serviceModule.py',
               'mosaic/lib/execPipe.py'],
    url = 'https://llrogcid.in2p3.fr/the-mosaic-framework/',
    description = 'Mosaic is a framework dedicated to the comparison of AI models.',
    long_description=long_description,
    long_description_content_type="text/markdown",
    packages = ['mosaic', 'mosaic.bin', 'mosaic.lib', 'mosaic.share'],
    install_requires=[
        'numpy',
        'rpyc',
        'matplotlib',
        'tqdm',
        'torch',
        'pandas',
        'seaborn'
    ],
    classifiers=[
        "Programming Language :: Python :: 3.8",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    include_package_data=True,
    python_requires='>=3.8'
)
