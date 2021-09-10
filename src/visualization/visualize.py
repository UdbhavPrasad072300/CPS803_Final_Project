import matplotlib.pyplot as plt


def plot_sequential(sequence, x_label, y_label):
    plt.plot(sequence)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.show()
