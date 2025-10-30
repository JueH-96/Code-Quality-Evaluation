# Read the input
N = int(input())
P = list(map(int, input().split()))
Q = list(map(int, input().split()))

# Compute the answer
S = []
for i in range(N):
    S.append(Q[P[i]-1])

# Print the output
print(" ".join(map(str, S)))