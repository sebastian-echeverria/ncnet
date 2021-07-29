#!/usr/bin/env bash
docker run --rm --mount type=bind,source="$(pwd)"/../datasets,target=/datasets \
                aimls/coords $@
