from mnist import MNIST
import numpy as np
import cv2

class Loader:
	def __init__(self):
		self.mndata = MNIST('./assets')
		self.mndata.gz = True
		self.train_data, self.train_labels = self.mndata.load_training()
		self.test_data, self.test_labels = self.mndata.load_testing()

		# Numpy array of size [60000, 784]
		self.train_data = np.array(self.train_data).astype(np.uint8)

		# Numpy array of size [10000, 784]
		self.test_data = np.array(self.test_data).astype(np.uint8)

		# Numpy array of size [60000,]
		self.train_labels = np.array(self.train_labels).astype(np.int32)

		# Numpy array of size [10000,]
		self.test_labels = np.array(self.test_labels).astype(np.int32)

	# Shows a preview of 16 randomly selected images from the train dataset along with the labels
	def preview(self):
		indices = np.random.randint(0, self.test_data.shape[0], (4,4))
		images = [ [ self.test_data[indices[i,j]].reshape((28,28)) for j in range(4)] for i in range(4) ]

		final = np.full((28 * 4 + 4 * 20, 28 * 4 + 3 * 5),255,np.uint8)
		for i in range(4):
			for j in range(4):
				final[j*(28+20):j*(28+20)+28,i*(28+5):i*(28+5)+28] = images[i][j]
				final = cv2.putText(final,
					str(self.test_labels[indices[i,j]]),
					(i*(28+5), (j+1)*(28+18)),
					cv2.FONT_HERSHEY_COMPLEX,
					0.5,
					(0,0,0),
					1,
					cv2.LINE_AA)
		
		final = cv2.resize(final, (0,0), fx=2, fy=2)
		cv2.imshow('Preview', final)
		cv2.waitKey(0)
		cv2.destroyAllWindows()