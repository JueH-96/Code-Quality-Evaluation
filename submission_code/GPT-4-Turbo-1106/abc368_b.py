# Read the number of elements
N = int(input())

# Read the elements of the array
A = list(map(int, input().split()))

# Initialize the counter for the operations
operations = 0

# Repeat the operation until A contains one or fewer positive elements
while len([x for x in A if x > 0]) > 1:
    # Sort A in descending order
    A.sort(reverse=True)
    # Decrease both A_1 and A_2 by 1
    A[0] -= 1
    A[1] -= 1
    # Increment the counter
    operations += 1

# Print the number of operations
print(operations)