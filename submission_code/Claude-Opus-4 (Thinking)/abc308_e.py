# YOUR CODE HERE
def mex(a, b, c):
    values = {a, b, c}
    for i in range(4):
        if i not in values:
            return i
    return 3

N = int(input())
A = list(map(int, input().split()))
S = input().strip()

# Precompute prefix counts of M's by value
prefix_M = [[0, 0, 0] for _ in range(N+1)]
for i in range(N):
    for v in range(3):
        prefix_M[i+1][v] = prefix_M[i][v]
    if S[i] == 'M':
        prefix_M[i+1][A[i]] += 1

# Precompute suffix counts of X's by value
suffix_X = [[0, 0, 0] for _ in range(N+1)]
for i in range(N-1, -1, -1):
    for v in range(3):
        suffix_X[i][v] = suffix_X[i+1][v]
    if S[i] == 'X':
        suffix_X[i][A[i]] += 1

total = 0

# For each position j where S[j] = 'E'
for j in range(N):
    if S[j] == 'E':
        v = A[j]
        
        # Calculate contribution
        for a in range(3):
            for c in range(3):
                total += prefix_M[j][a] * suffix_X[j+1][c] * mex(a, v, c)

print(total)