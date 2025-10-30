# YOUR CODE HERE
input_sequence = list(map(int, input().split()))

result = sum(val * (2 ** idx) for idx, val in enumerate(reversed(input_sequence)))

print(result)