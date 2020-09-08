# Rabbit

Rabbit is an abstract interpretation library for Rust. This library is
currently very much a work in progress, and for now if you need an abstract
interpretation engine you are better off using
[Apron](https://github.com/antoinemine/apron) or
[ELINA](https://github.com/eth-sri/ELINA). There are several issues that need
to be addressed before rabbit is ready to be used:

* I currently use floating point values to represent real numbers, but no
  special consideration is given to rounding issues. This can lead to
  unsoundness.
* No particular attention has been paid to efficiency yet so this library is
  likely to be very slow compared to existing abstract interpretation engines.

There are also several features I plan to add over time:

* New numerical abstract domains including zonotopes, ellipsoids, and octagons.
* More utilities for non-numerical domains, such as a predicate abstraction.

# Dependencies

Rabbit uses [GLPK](https://www.gnu.org/software/glpk/) for solving linear
programming problems, and requires the GLPK header and library files to be
available. On Ubuntu-like systems GLPK can be installed from the package
manager. Additionally, bindings to GLPK are generated using
[bindgen](https://github.com/rust-lang/rust-bindgen). Internally, bindgen
relies on the Clang libraries to parse and and preprocess C files. (This is
important because it allows bindings to be correctly generated on-the-fly for
any platform). Therefore you will need clang development files in order to use
Rabbit. On Ubuntu-like systems all of the necessary dependencies can be
installed with

    sudo apt install glpk llvm-dev libclang-dev clang
