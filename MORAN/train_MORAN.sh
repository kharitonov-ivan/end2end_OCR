GPU=0
CUDA_VISIBLE_DEVICES=${GPU} \
python main.py \
	--train_coco /srv/hd3/data/izakharkin/COCO/lmdb/COCO-Text-train.lmdb \
	--valroot /srv/hd3/data/izakharkin/COCO/lmdb/COCO-Text-val.lmdb \
	--workers 2 \
	--batchSize 16 \
	--niter 10 \
	--lr 1 \
	--experiment output_cocotext17/ \
	--displayInterval 100 \
	--valInterval 1000 \
	--saveInterval 40000 \
	--adadelta \
	--BidirDecoder \
	--cuda \
	--MORAN demo.pth
