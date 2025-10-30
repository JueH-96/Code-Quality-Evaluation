N = int(input())
S = []
for _ in range(N):
    S.append(input())

# Find maximum length
M = max(len(s) for s in S)

# Create empty result grid
T = [['*'] * N for _ in range(M)]

# Fill in characters from input strings
for i in range(N):
    for j in range(len(S[i])):
        T[j][N-i-1] = S[i][j]

# Generate output strings by removing trailing asterisks
result = []
for row in T:
    # Convert row to string
    s = ''.join(row)
    # Remove trailing asterisks but keep internal ones
    while s.endswith('*'):
        s = s[:-1]
    result.append(s)

# Print result
for line in result:
    print(line)