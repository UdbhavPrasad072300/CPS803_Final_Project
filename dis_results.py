import matplotlib.pyplot as plt


plt.plot([1, 2, 3], [52.22, 57.96, 58.06], color="black", label="ViT - No KD")
plt.plot([1, 2, 3], [51.54, 59.32, 59.32], color="lightblue", label="ViT with ResNet")
plt.plot([1, 2, 3], [52.34, 57.78, 55.64], color="blue", label="DeiT with ResNet")
plt.plot([1, 2, 3], [51.28, 58.42, 58.22], color="lightsalmon", label="ViT with VGG")
plt.plot([1, 2, 3], [52.26, 57.76, 58.12], color="red", label="DeiT with VGG")
plt.legend()
plt.title("Hard Label Distillation")
plt.xlabel("Num. of Encoders")
plt.xticks([1, 2, 3])
plt.ylabel("Accuracy")
plt.ylim(0, 70)
plt.show()


plt.plot([1, 2, 3], [52.22, 57.96, 58.06], color="black", label="ViT - No KD")
plt.plot([1, 2, 3], [36.4, 50.82, 55.74], color="lightblue", label="ViT with ResNet")
plt.plot([1, 2, 3], [38.1, 51.66, 55.22], color="blue", label="DeiT with ResNet")
plt.plot([1, 2, 3], [43.9, 56.46, 54.94], color="lightsalmon", label="ViT with VGG")
plt.plot([1, 2, 3], [45.26, 54.02, 56.52], color="red", label="DeiT with VGG")
plt.legend()
plt.title("Soft Label Distillation")
plt.xlabel("Num. of Encoders")
plt.xticks([1, 2, 3])
plt.ylabel("Accuracy")
plt.ylim(0, 70)
plt.show()
