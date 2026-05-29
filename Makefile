.PHONY: install, clean, check_conda, upload_pypi

install: check_conda
	wget -nc --content-disposition "https://zenodo.org/record/14756042/files/data_archive.zip?download=1" -P ./experiments/datasets
	wget -nc --content-disposition "https://zenodo.org/record/14764011/files/embs_archive.zip?download=1" -P ./experiments/embs
	wget -nc --content-disposition "https://zenodo.org/records/17672438/files/predictions.zip?download=1" -P ./experiments/
	python -c "import zipfile; zipfile.ZipFile('./experiments/datasets/data_archive.zip').extractall('./experiments/datasets/')"
	python -c "import zipfile; zipfile.ZipFile('./experiments/embs/embs_archive.zip').extractall('./experiments/embs/')"
	python -c "import zipfile; zipfile.ZipFile('./experiments/predictions.zip').extractall('./experiments/')"
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

upload_pypi:
	rm -rf build/ dist/
	python setup.py sdist bdist_wheel
	twine upload dist/* -u __token__ -p $(PYPI_API_TOKEN)