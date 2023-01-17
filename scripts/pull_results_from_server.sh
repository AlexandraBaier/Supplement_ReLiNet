server_path=arboghast.baierax:/workspace/Supplement_ReLiNet

mkdir datasets
mkdir results
mkdir models
mkdir environment

rsync -r --ignore-existing  $server_path/datasets/ datasets
rsync -r --ignore-existing  $server_path/results/ results
rsync -r --ignore-existing  $server_path/models/ models
rsync -r --ignore-existing  $server_path/environment/ environment
rsync -r --ignore-existing  $server_path/configuration/ configuration