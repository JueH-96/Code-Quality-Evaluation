# Read the first line and split into N and K, convert to integers
N_K = input().split()
N = int(N_K[0])
K = int(N_K[1])

# Read the second line and split into list of integers A
A = list(map(int, input().split()))

# Perform the operation: take the last K elements and move them to the front
new_list = A[-K:] + A[:-K]

# Print the new list as space-separated strings
print(' '.join(map(str, new_list)))