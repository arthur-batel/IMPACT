.PHONY: install, clean, check_conda

install: check_conda
	wget -nc --content-disposition "https://zenodo.org/record/14756042/files/data_archive.zip?download=1" -P ./experiments/datasets
	wget -nc --content-disposition "https://zenodo.org/record/14764011/files/embs_archive.zip?download=1" -P ./experiments/embs
	unzip -n ./experiments/datasets/data_archive.zip -d ./experiments/datasets/
	unzip -n ./experiments/embs/embs_archive.zip -d ./experiments/embs/
	conda init
	conda env create -f environment.yaml -n impact-env \
  || conda env update -f environment.yaml -n impact-env;

clean:
	rm -rf data/
	rm -rf results/

check_conda:
	@if command -v conda >/dev/null 2>&1; then \
		echo "conda is installed"; \
	else \
		echo "conda needs to be installed\nrun the makefile again after the installation"; \
		exit 1; \
	fi