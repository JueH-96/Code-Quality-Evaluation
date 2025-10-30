# YOUR CODE HERE

# Read the inputs
N = int(input())
A = list(map(int, input().split()))

# Sort the list in descending order
A.sort(reverse=True)

# Initialize the operation counter
operations = 0

# Perform the operation until A contains one or fewer positive elements
while len([x for x in A if x > 0]) > 1:
    A[0] -= 1
    A[1] -= 1
    operations += 1
    A.sort(reverse=True)

# Print the number of operations
print(operations)