pip install openai pronouncing epitran langdetect spacy beautifulsoup4 nltk toml datasets --index-url https://pypi.org/simple
pip install aws_bedrock_token_generator --index-url https://pypi.org/simple

cd ${HOME:-/tmp} || exit
git clone https://github.com/festvox/flite.git
cd flite/ || exit
sh configure && make
make install
cd testsuite  || exit
make lex_lookup
cp lex_lookup /usr/local/bin

python -m spacy download en_core_web_sm
python -m nltk.downloader averaged_perceptron_tagger_eng
python -m nltk.downloader punkt
python -m nltk.downloader punkt_tab
python -m nltk.downloader cmudict
python -m nltk.downloader wordnet
