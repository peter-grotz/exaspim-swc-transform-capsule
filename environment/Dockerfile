# hash:sha256:0000000000000000000000000000000000000000000000000000000000000000
ARG REGISTRY_HOST
FROM $REGISTRY_HOST/codeocean/mambaforge3:24.5.0-0-python3.12.4-ubuntu22.04

ARG DEBIAN_FRONTEND=noninteractive
ARG GIT_ASKPASS
COPY git-ask-pass /

# This stage runs on Python 3.11, not the base image's 3.12.
#
# allensdk 2.16.2 -- the newest release, and a hard dependency of
# aind_exaspim_register_cells -- caps numpy<1.24, and numpy 1.23.5 cannot build on 3.12
# because it needs distutils, removed from the stdlib there. Upstream fixed this in
# AllenInstitute/AllenSDK#2769 but has not released since Nov 2023.
#
# The base tag is stepped down rather than chosen directly because the only mambaforge3
# tags observed in this deployment are python3.10.12 and python3.12.4. If a 3.11 tag
# exists, use it directly and drop this line.
RUN mamba install -y python=3.11 && mamba clean -afy

RUN pip install -U --no-cache-dir \
    "git+https://github.com/AllenNeuralDynamics/aind_exaspim_register_cells.git" \
    # Pinned to a commit, not a branch: Docker keys this layer on the text of the
    # line, so '@dev' reuses a cached layer and silently installs nothing new.
    # Bump this SHA whenever the library changes.
    "exaspim-swc-processing @ git+https://github.com/peter-grotz/exaspim-swc-processing.git@dbfbb6543421f4271210067dd3e16514c296dae5"
