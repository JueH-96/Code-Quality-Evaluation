import itertools

# Read the number of functions N and the length of sequence K
N, K = map(int, input().split())

# Read the coefficients of the linear functions
functions = [tuple(map(int, input().split())) for _ in range(N)]

# Initialize the maximum value to a very small number
max_value = float('-inf')

# Generate all possible sequences of K distinct integers between 1 and N
for sequence in itertools.permutations(range(N), K):
    # Start with the initial value of x as 1
    x = 1
    # Apply the functions in the sequence order
    for index in sequence:
        A, B = functions[index]
        x = A * x + B
    # Update the maximum value if the current sequence gives a higher value
    max_value = max(max_value, x)

# Print the maximum value
print(max_value)