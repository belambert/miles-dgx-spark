# miles-run

Non-Docker install of [Miles](https://github.com/radixark/miles) for a DGX Spark (GB10, aarch64, CUDA 13).

## Usage

The installer clones the sources into `./miles-src` and builds a venv in `./.venv`. It needs the pinned `radixark/miles` Docker image pulled locally, because a few compiled packages are copied out of it.

    ./install.sh 2>&1 | tee install.log

Check an existing install:

    ./verify.sh

Set `PREFIX` or `SRC` to use other locations. The pins at the top of `install.sh` say which image and commits they match.
