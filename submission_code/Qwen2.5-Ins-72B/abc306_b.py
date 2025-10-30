# YOUR CODE HERE
input_sequence = list(map(int, input().split()))
result = 0
for i, value in enumerate(input_sequence):
    result += value * (2 ** i)
print(result)