# YOUR CODE HERE
sequence = input().split()
result = sum(int(bit) * (2 ** i) for i, bit in enumerate(sequence))
print(result)