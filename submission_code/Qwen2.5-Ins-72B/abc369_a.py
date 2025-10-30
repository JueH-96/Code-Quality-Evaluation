# YOUR CODE HERE
A, B = map(int, input().split())

# Function to check if three numbers form an arithmetic sequence
def is_arithmetic_sequence(p, q, r):
    return q - p == r - q

# Count the number of valid x
count = 0

# Check for x such that (A, B, x) can form an arithmetic sequence
for x in range(1, 201):  # x can be in the range of 1 to 200 to cover all possible values
    if is_arithmetic_sequence(A, B, x) or is_arithmetic_sequence(A, x, B) or is_arithmetic_sequence(B, A, x) or is_arithmetic_sequence(B, x, A) or is_arithmetic_sequence(x, A, B) or is_arithmetic_sequence(x, B, A):
        count += 1

# Check for negative x
for x in range(-200, 0):  # x can be in the range of -200 to -1 to cover all possible values
    if is_arithmetic_sequence(A, B, x) or is_arithmetic_sequence(A, x, B) or is_arithmetic_sequence(B, A, x) or is_arithmetic_sequence(B, x, A) or is_arithmetic_sequence(x, A, B) or is_arithmetic_sequence(x, B, A):
        count += 1

print(count)