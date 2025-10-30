def min_operations_to_match_sequences(N, M, A, B):
    def distance(a, b, M):
        return min((a - b) % M, (b - a) % M)

    operations = 0
    for i in range(N):
        if A[i] == B[i]:
            continue
        if i > 0 and (A[i - 1] == B[i] or A[i] == B[i - 1]):
            return -1
        if i < N - 1 and (A[i + 1] == B[i] or A[i] == B[i + 1]):
            return -1
        operations += distance(A[i], B[i], M)
    return operations

# Read input from stdin
N, M = map(int, input().split())
A = list(map(int, input().split()))
B = list(map(int, input().split()))

# Solve the problem
result = min_operations_to_match_sequences(N, M, A, B)

# Write the answer to stdout
print(result)