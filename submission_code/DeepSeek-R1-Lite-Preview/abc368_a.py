# Read N and K from the first line
N_K = input().split()
N = int(N_K[0])
K = int(N_K[1])

# Read the list of A_i from the second line
A = list(map(int, input().split()))

# Perform the operation: take last K elements and place them on top
B = A[N-K:] + A[:N-K]

# Print the result as space-separated integers
print(' '.join(map(str, B)))