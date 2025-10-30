# Read N, K, X from input
N_K_X = input().split()
N = int(N_K_X[0])
K = int(N_K_X[1])
X = int(N_K_X[2])

# Read the sequence A
A = list(map(int, input().split()))

# Insert X at position K (after the K-th element)
A.insert(K, X)

# Print the new sequence
print(' '.join(map(str, A)))