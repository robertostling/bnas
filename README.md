# BNAS -- Basic Neural Architecture Subprograms

The goal of this project is to provide a set of helper classes and functions
for [Theano](https://github.com/Theano/Theano) to facilitate experiments with
machine learning, in particular for natural language processing.
Unlike libraries such as [Keras](http://keras.io/) and
[Blocks](https://github.com/mila-udem/blocks), the purpose is rather to help
people (myself in particular) use Theano efficiently when developing new
architectures.

BNAS is used by and developed in tandem with
[HNMT](https://github.com/robertostling/hnmt), the Helsinki Neural Machine
Translation (NMT) system. A simpler NMT system for educational purposes can
be found in the `examples/` directory.

## Installing

Since this is a project currently under development, you probably want to
install it in development mode:
```
python3 setup.py develop --user
```

## Documentation

The `examples/` directory contains commented example code that you could start
from.

Install Sphinx and run `docs/build.sh` to build the API reference.

