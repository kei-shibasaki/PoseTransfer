train_eval-%:
	$(eval MODEL_NAME = "fashion_large_mod")
	$(eval CONFIG_FILE = "config/config_${MODEL_NAME}.json")
	$(eval LOGDIR_PRE = "results/${MODEL_NAME}_pre")
	$(eval LOGDIR_FINE = "results/${MODEL_NAME}_fine")
	
	CUDA_VISIBLE_DEVICES=${@:train-%=%} python -m train_codes.train_pre --config ${CONFIG_FILE}
	CUDA_VISIBLE_DEVICES=${@:train-%=%} python -m train_codes.train_fine --config ${CONFIG_FILE}

	CUDA_VISIBLE_DEVICES=${@:train-%=%} python -m eval_codes.generate_images -b 16 -m pre --config ${CONFIG_FILE}
	CUDA_VISIBLE_DEVICES=${@:train-%=%} python -m eval_codes.evaluation -m pre --config ${CONFIG_FILE}

	CUDA_VISIBLE_DEVICES=${@:train-%=%} python -m eval_codes.generate_images -b 16 -m fine --config ${CONFIG_FILE}
	CUDA_VISIBLE_DEVICES=${@:train-%=%} python -m eval_codes.evaluation -m fine --config ${CONFIG_FILE}

	echo >> ${LOGDIR_PRE}/results.txt
	CUDA_VISIBLE_DEVICES=${@:train-%=%} python -m pytorch_fid ${LOGDIR_PRE}/GT ${LOGDIR_PRE}/generated --device cuda:0 >> ${LOGDIR_PRE}/results.txt

	echo >> ${LOGDIR_FINE}/results.txt
	CUDA_VISIBLE_DEVICES=${@:train-%=%} python -m pytorch_fid ${LOGDIR_FINE}/GT ${LOGDIR_FINE}/generated --device cuda:0 >> ${LOGDIR_FINE}/results.txt

eval-%:
	$(eval MODEL_NAME = "fashion_large_mod")
	$(eval CONFIG_FILE = "config/config_${MODEL_NAME}.json")
	$(eval LOGDIR_PRE = "results/${MODEL_NAME}_pre")
	$(eval LOGDIR_FINE = "results/${MODEL_NAME}_fine")

	CUDA_VISIBLE_DEVICES=${@:train-%=%} python -m eval_codes.generate_images -b 16 -m pre --config ${CONFIG_FILE}
	CUDA_VISIBLE_DEVICES=${@:train-%=%} python -m eval_codes.evaluation -m pre --config ${CONFIG_FILE}

	CUDA_VISIBLE_DEVICES=${@:train-%=%} python -m eval_codes.generate_images -b 16 -m fine --config ${CONFIG_FILE}
	CUDA_VISIBLE_DEVICES=${@:train-%=%} python -m eval_codes.evaluation -m fine --config ${CONFIG_FILE}

	echo >> ${LOGDIR_PRE}/results.txt
	CUDA_VISIBLE_DEVICES=${@:train-%=%} python -m pytorch_fid ${LOGDIR_PRE}/GT ${LOGDIR_PRE}/generated --device cuda:0 >> ${LOGDIR_PRE}/results.txt

	echo >> ${LOGDIR_FINE}/results.txt
	CUDA_VISIBLE_DEVICES=${@:train-%=%} python -m pytorch_fid ${LOGDIR_FINE}/GT ${LOGDIR_FINE}/generated --device cuda:0 >> ${LOGDIR_FINE}/results.txt