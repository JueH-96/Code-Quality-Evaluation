# YOUR CODE HERE

# Read the input from stdin
input_sequence = input().split()

# Convert the input sequence to integers
input_sequence = [int(i) for i in input_sequence]

# Calculate the sum
sum = 0
for i in range(64):
    sum += input_sequence[i] * (2 ** i)

# Print the sum
print(sum)