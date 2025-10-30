# Read the input
N, M = map(int, input().split())
S = input()
C = list(map(int, input().split()))

# Perform the circular shift operations
for i in range(1, M+1):
    indices = [j for j in range(N) if C[j] == i]
    S = S[:indices[0]] + S[indices[-1]] + S[indices[0]+1:indices[-1]] + S[indices[0]]

# Print the final string
print(S)