# Importing an ML library (PyTorch)
import torch

# Other Imports
import argparse
import random
import numpy as np
from dataLoader import Loader
import os
import cv2

# Custom Modules
import torchvision
from torchvision import datasets, transforms
from torch import nn, optim

# Extra modules for testing and other purposes
import matplotlib.pyplot as plt
from time import time

# Utilities
def unison_shuffled_copies(a, b):
	assert len(a) == len(b)
	p = np.random.permutation(len(a))
	return a[p], b[p]

def max_ind(input_array):
	input_size = len(input_array)
	max_val = input_array[0]
	ret_ind = 0
	if input_size == 1:
		return ret_ind
	i = 1
	while i < input_size:
		if input_array[i] > max_val:
			max_val = input_array[i]
			ret_ind = i
		i = i + 1
	return ret_ind

# This is the class for training our model
class Trainer:
	def __init__(self):

		# Seeding the RNG's
		# This is the point where we seed our ML library
		np.random.seed(12345)
		random.seed(12345)
		torch.manual_seed(12345)

		# Setting hyperparameters.
		self.batch_size = 64 # Batch Size
		self.num_epochs = 20 # Number of Epochs to train for
		self.lr = 0.001       # Learning rate
		self.input_size = 784
		self.hl1_size = 128
		self.hl2_size = 64
		self.hidden_sizes = [self.hl1_size, self.hl2_size]
		self.output_size = 10

		# Initializing the model, loss, optimizer, etc
		self.model = nn.Sequential(nn.Linear(self.input_size, self.hidden_sizes[0]), nn.ReLU(), nn.Linear(self.hidden_sizes[0], self.hidden_sizes[1]), nn.ReLU(), nn.Linear(self.hidden_sizes[1], self.output_size))
		self.loss = nn.CrossEntropyLoss()
		# self.loss = nn.NLLLoss()
		self.optimizer = optim.SGD(self.model.parameters(), lr=self.lr, momentum=0.9)

	def load_data(self):
		# Loading the Data
		self.loader = Loader()

		# Changing Data into representation favored by ML library (torch.Tensor for pytorch)
		# transforms.ToTensor(),
		# transform = transforms.Compose([transforms.Normalize((0.5,), (0.5,))])
		train_data = torch.from_numpy(self.loader.train_data.astype(np.float32))
		self.train_data = train_data/255.0
		self.train_labels = torch.from_numpy(self.loader.train_labels)
		test_data = torch.from_numpy(self.loader.test_data.astype(np.float32))
		self.test_data = test_data/255.0
		self.test_labels = torch.from_numpy(self.loader.test_labels)
		print(self.train_data.shape)
		print(self.train_labels.shape)

	def save_model(self):
		# Saving the model parameters into the file 'assets/model'
		# For pytorch, torch.save(self.model.state_dict(), 'assets/model')
		torch.save(self.model.state_dict(), 'assets/model')

	def load_model(self):
		# Loading the model parameters from the file 'assets/model'
		if os.path.exists('assets/model'):
			# For pytorch, self.model.load_state_dict(torch.load('assets/model'))
			self.model.load_state_dict(torch.load('assets/model'))
			pass
		else:
			raise Exception('Model not trained')

	def train(self):
		if not self.model:
			return

		print("Training...")
		for epoch in range(self.num_epochs):
			train_loss = self.run_epoch()

			self.save_model()

			print(f'	Epoch #{epoch+1} trained')
			print(f'		Train loss: {train_loss:.3f}')
		print('Training Complete')

	def test(self):
		if not self.model:
			return 0

		print(f'Running test...')
		# Initializing running loss
		running_loss = 0.0

		# Setting the ML library to freeze the parameter training

		with torch.no_grad():

			i = 0 # Number of batches
			correct = 0 # Number of correct predictions
			for batch in range(0, self.test_data.shape[0], self.batch_size):
				batch_X = self.test_data[batch: batch+self.batch_size] # shape [batch_size,784] or [batch_size,28,28]
				batch_Y = self.test_labels[batch: batch+self.batch_size] # shape [batch_size,]
				
				# Finding the predictions
				predict_Y = self.model(batch_X)
				predict_Y = predict_Y.type(torch.DoubleTensor)

				# Finding the loss
				batch_Y = batch_Y.type(torch.LongTensor)
				loss = self.loss(predict_Y, batch_Y)
				running_loss = running_loss + loss
				# predict_Y = predict_Y.type(torch.LongTensor)
				# Finding the number of correct predictions and update correct
				j = 0
				for prediction in predict_Y:
					b = batch_Y[j]
					predic = max_ind(prediction)
					if predic == batch_Y[j]:
						correct = correct + 1
					j = j + 1
				# print(predict_Y.shape)
				# print(batch_Y.shape)
				# comp_Y = torch.eq(predict_Y, batch_Y)
				# for comp in comp_Y:
					# if comp == True:
						# count = count + 1
				
				i += 1
		
		print(f'	Test loss: {(running_loss/i):.3f}')
		print(f'	Test accuracy: {(correct*100/self.test_data.shape[0]):.2f}%')

		return correct/self.test_data.shape[0]

	def run_epoch(self):
		# Initializing running loss
		running_loss = 0.0

		# Setting the ML library to enable the parameter training

		# Shuffling the data
		self.train_data, self.train_labels = unison_shuffled_copies(self.train_data, self.train_labels)
		
		i = 0 # Number of batches
		for batch in range(0, self.train_data.shape[0], self.batch_size):
			batch_X = self.train_data[batch: batch+self.batch_size] # shape [batch_size,784] or [batch_size,28,28]
			batch_Y = self.train_labels[batch: batch+self.batch_size] # shape [batch_size,]


			# Finding the predictions
			predict_Y = self.model(batch_X)
			predict_Y = predict_Y.type(torch.DoubleTensor)
			# predict_Y.next()
			# print(batch_X.shape)
			# print(predict_Y.shape)
			# print(batch_Y.shape)

			

			# Finding the loss
			batch_Y = batch_Y.type(torch.LongTensor)
			loss = self.loss(predict_Y, batch_Y)
			running_loss = running_loss + loss

			self.optimizer.zero_grad()

			# Backpropagation
			loss.backward()
			self.optimizer.step()

			# Updating the running loss
			i += 1
		
		return running_loss / i

	def predict(self, image):
		prediction = 0
		if not self.model:
			return prediction

		# Changing image into representation favored by ML library (eg torch.Tensor for pytorch)
		# transform = transforms.Compose([transforms.ToTensor(),transforms.Normalize((0.5,), (0.5,)),])
		# img = transform(image)
		img = torch.from_numpy(image.astype(np.float32))
		img = img/255.0
		
		# Predicting the digit value using the model
		prediction = self.model(img)
		ret_val = max_ind(prediction)

		return ret_val

if __name__ == '__main__':
	parser = argparse.ArgumentParser(description='Model Trainer')
	parser.add_argument('-train', action='store_true', help='Train the model')
	parser.add_argument('-test', action='store_true', help='Test the trained model')
	parser.add_argument('-preview', action='store_true', help='Show a preview of the loaded test images and their corresponding labels')
	parser.add_argument('-predict', action='store_true', help='Make a prediction on a randomly selected test image')

	options = parser.parse_args()

	t = Trainer()
	if options.train:
		t.load_data()
		t.train()
		t.test()
	if options.test:
		t.load_data()
		t.load_model()
		t.test()
	if options.preview:
		t.load_data()
		t.loader.preview()
	if options.predict:
		t.load_data()
		try:
			t.load_model()
		except:
			pass
		i = np.random.randint(0,t.loader.test_data.shape[0])

		print(f'Predicted: {t.predict(t.loader.test_data[i])}')
		print(f'Actual: {t.loader.test_labels[i]}')

		image = t.loader.test_data[i].reshape((28,28))
		image = cv2.resize(image, (0,0), fx=16, fy=16)
		cv2.imshow('Digit', image)
		cv2.waitKey(0)
		cv2.destroyAllWindows()