from model import Trainer

def testModel():
	t = Trainer()
	t.load_data()

	try:
		t.load_model()
	except:
		t.train()
	
	acc = t.test()

	assert acc > 0.9, "Accuracy not high enough"