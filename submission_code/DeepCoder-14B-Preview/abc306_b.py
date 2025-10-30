# Read the input as a list of integers
bits = list(map(int, input().split()))

# Calculate the total by summing each bit multiplied by 2^i
total = 0
for i in range(64):
    total += bits[i] * (2 ** i)

# Print the result
print(total)