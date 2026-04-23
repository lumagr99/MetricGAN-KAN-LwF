
import os


def _expand(path):
	return os.path.expanduser(path)


_DEFAULT_CHIME_DATA_ROOT = '/mnt/parscratch/users/acp20glc/CHiME2023/data'
_CHIME_DATA_ROOT = _expand(os.environ.get('CHIME2023_DATA_ROOT', _DEFAULT_CHIME_DATA_ROOT))

LIBRI3MIX_ROOT_PATH = _expand(
	os.environ.get(
		'LIBRI3MIX_ROOT_PATH',
		os.path.join(_CHIME_DATA_ROOT, 'LibriMix', 'Libri3Mix'),
	)
)
CHiME_ROOT_PATH = _expand(
	os.environ.get(
		'CHIME_ROOT_PATH',
		os.path.join(_CHIME_DATA_ROOT, 'CHiME-5', 'CHiME5_processed'),
	)
)
LIBRICHiME_ROOT_PATH = _expand(
	os.environ.get(
		'LIBRICHIME_ROOT_PATH',
		os.path.join(_CHIME_DATA_ROOT, 'reverberant-LibriCHiME-5', 'data'),
	)
)

API_KEY = '2cpRt3u7Q77WnbUPcgygoCZL6'
