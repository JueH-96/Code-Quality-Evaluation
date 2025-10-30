# YOUR CODE HERE

# Read the inputs
N = int(input().strip())
A = list(map(int, input().strip().split()))

# Sort the sequence in descending order
A.sort(reverse=True)

# Initialize the counter
counter = 0

# Perform the operations until A contains one or fewer positive elements
while len(A) > 1:
    # Decrease both A_1 and A_2 by 1
    A[0] -= 1
    A[1] -= 1
    # Sort the sequence in descending order
    A.sort(reverse=True)
    # Increment the counter
    counter += 1

# Print the number of times he performs the operation
print(counter)