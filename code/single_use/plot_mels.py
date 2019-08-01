import matplotlib.pyplot as plt
import numpy as np


for i in range(1):
	a = np.load('../eval/mels_save/{}_mel.npy'.format(i))
	b = np.load('../eval/mels_save/{}_ref_emt.npy'.format(i))
	c = np.load('../eval/mels_save/{}_ref_spk.npy'.format(i))
	plt.figure()
	plt.title(str(i) + '_' + 'mel')
	plt.imshow(a[:500,:].T,origin='lower')
	plt.figure()
	plt.title(str(i) + '_' + 'emt')
	plt.imshow(b[:500, :].T, origin='lower')
	plt.figure()
	plt.title(str(i) + '_' + 'spk')
	plt.imshow(c[:500, :].T, origin='lower')


plt.show()