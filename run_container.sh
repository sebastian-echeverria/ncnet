 #!/usr/bin/env bash
 docker run --rm --gpus=all --mount type=bind,source="$(pwd)"/datasets,target=/app/datasets \
                            --mount type=bind,source=$HOME/.torch/models,target=/root/.torch/models \
                            ncnet/ncnet-uav
