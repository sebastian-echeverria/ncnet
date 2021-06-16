 #!/usr/bin/env bash
 docker run --mount type=bind,source="$(pwd)"/datasets,target=/app/datasets ncnet/ncnet-uav
