module load gcccore/13.3.0 python/3.12.3 cuda/12.4.0

if [ ! -d "ENV" ]; then
	python -m venv ENV
fi

source ENV/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
pip install flash-attn --no-build-isolation

