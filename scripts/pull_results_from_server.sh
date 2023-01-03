server_path=arboghast.baierax:/workspace/Supplement_ReLiNet

mkdir datasets
mkdir results
mkdir models
mkdir environment

rsync -r $server_path/datasets/ datasets
rsync -r $server_path/results/ results
rsync -r $server_path/models/ models
rsync -r $server_path/environment/ environment
rsync -r $server_path/configuration/ configuration