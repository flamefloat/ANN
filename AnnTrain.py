import numpy as np 
import Ann
import matplotlib.pyplot as plt

EPOCHS = 1000
INPUT_SIZE = 7
HIDDEN_SIZE1 = 30 
HIDDEN_SIZE2 = 30
OUTPUT_SIZE = 1
learningRate = 0.1

start, end = 0 * np.pi, 2 * np.pi
TIME_STEP = 30
steps = np.linspace(start, end, TIME_STEP, dtype=np.float32)
temp_inputs = np.sin(steps)
inputs = np.zeros((24,7))
for i in range(24):
    inputs[i,:] = temp_inputs[i:i+7]
label = np.cos(steps[3:27])
inputs = inputs.reshape(24,1,7)
label = label.reshape(24,1,1)


def train(ann, inputs, label):
    for i in range(EPOCHS):
        loss = 0
        out = np.zeros(24)
        for j in range(inputs.shape[0]):
            out[j] = ann.forward(inputs[j,:])
            ann.backward(label[j,:])
            loss += (label[j,:] - out[j])**2
        loss = loss/inputs.shape[0]
        if loss < 0.01:
            break
        if i % 100 == 0:
            print('EPOCHS:',i,'Loss:',loss)
    return out

if __name__ =='__main__':
    myAnn = Ann.ANN(INPUT_SIZE, HIDDEN_SIZE1, HIDDEN_SIZE2, OUTPUT_SIZE, learningRate)
    out = train(myAnn, inputs, label)
    plt.plot(steps[3:27], out.flatten(), 'r-', label = 'predict_data')
    plt.plot(steps[3:27], label.flatten(), 'b-', label = 'label')
    plt.legend()
    plt.title('Use sin(x) predict cos(x)')
    plt.show()

