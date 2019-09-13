from .tacotron import Tacotron
from .tacotron_emt_attn import Tacotron_emt_attn


def create_model(name, hparams):
  if name == 'Tacotron':
    return Tacotron(hparams)
  elif name == 'Tacotron_emt_attn':
    return Tacotron_emt_attn(hparams)
  else:
    raise Exception('Unknown model: ' + name)
