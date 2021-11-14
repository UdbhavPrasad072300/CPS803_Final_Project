import matplotlib.pyplot as plt


def plot_sequential(sequence, title, x_label, y_label):
    plt.plot(sequence)
    plt.title(title)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.show()
