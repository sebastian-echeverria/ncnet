 #!/usr/bin/env bash
 docker run --rm --gpus=all --mount type=bind,source="$(pwd)"/datasets,target=/app/datasets ncnet/ncnet-uav
