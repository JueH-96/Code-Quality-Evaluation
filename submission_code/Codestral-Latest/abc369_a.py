# YOUR CODE HERE
A, B = map(int, input().split())

# Function to check if three numbers form an arithmetic sequence
def is_arithmetic_sequence(a, b, c):
    return (b - a) == (c - b)

# Initialize the count of valid x
count = 0

# Check for all possible values of x from 1 to 100
for x in range(1, 101):
    if is_arithmetic_sequence(A, B, x) or is_arithmetic_sequence(A, x, B) or is_arithmetic_sequence(x, A, B):
        count += 1

# Print the count of valid x
print(count)