# Read the input from stdin
N, L = map(int, input().split())
A = list(map(int, input().split()))

# Initialize a counter for the number of people who passed
passed = 0

# Iterate over the scores
for score in A:
    # Check if the score is greater than or equal to the passing score
    if score >= L:
        # If it is, increment the counter
        passed += 1

# Print the number of people who passed
print(passed)