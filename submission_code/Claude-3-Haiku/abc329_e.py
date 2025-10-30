import sys

# Read input
N, M = map(int, input().split())
S = input()
T = input()

# Initialize X as a string of '#' characters
X = '#' * N

# Iterate through X and replace M consecutive characters with T if possible
for i in range(N - M + 1):
    if X[i:i+M] == '#' * M:
        X = X[:i] + T + X[i+M:]

# Check if X matches S
if X == S:
    print("Yes")
else:
    print("No")