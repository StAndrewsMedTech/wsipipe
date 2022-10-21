.. highlight:: shell

============
Installation
============


Stable release
--------------

To install wsipipe, run this command in your terminal:

.. code-block:: console

    $ pip install wsipipe

This is the preferred method to install wsipipe, as it will always install the most recent stable release.

If you don't have `pip`_ installed, this `Python installation guide`_ can guide
you through the process.

.. _pip: https://pip.pypa.io
.. _Python installation guide: http://docs.python-guide.org/en/latest/starting/installation/


From sources
------------

The sources for wsipipe can be downloaded from the `Github repo`_.

You can either clone the public repository:

.. code-block:: console

    $ git clone git://github.com/StAndrewsMedTech/wsipipe

Or download the `tarball`_:

.. code-block:: console

    $ curl -OJL https://github.com/StAndrewsMedTech/wsipipe/tarball/master

Once you have a copy of the source, you can install it with:

.. code-block:: console

    $ python setup.py install


.. _Github repo: https://github.com/StAndrewsMedTech/wsipipe
.. _tarball: https://github.com/StAndrewsMedTech/wsipipe/tarball/master


Openslide
---------

Wsipipe requires openslide this cannot be installed by pip.
Please follow the instructions on https://openslide.org/api/python/ for your operating system.

for example on ubuntu::
    
    apt install -y openslide-tools