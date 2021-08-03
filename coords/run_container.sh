#!/usr/bin/env bash
mkdir images
rm -f images/*
docker run --rm --mount type=bind,source="$(pwd)"/../datasets,target=/datasets \
                --mount type=bind,source="$(pwd)"/images,target=/app/images \
                aimls/coords "$@"
