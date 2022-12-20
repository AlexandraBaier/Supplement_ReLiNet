server_path=arboghast.baierax:/root/Supplement_Physics_Residual-Bounded_LSTM

mkdir ../datasets
mkdir ../results
mkdir ../models
mkdir ../environment

rsync -r $server_path/datasets/ ../datasets
rsync -r $server_path/results/ ../results
rsync -r $server_path/models/ ../models
rsync -r $server_path/environment/ ../environment
rsync -r $server_path/configuration/ ../configuration