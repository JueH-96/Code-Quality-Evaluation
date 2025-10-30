# Read the input line
input_line = input()

# Split the input into a list of strings and convert them to integers
A = list(map(int, input_line.split()))

# Initialize total sum and the power of 2
total = 0
power = 1

# Compute the sum A0*2^0 + A1*2^1 + ... + A63*2^63
for i in range(64):
    total += A[i] * power
    power *= 2

# Print the result
print(total)