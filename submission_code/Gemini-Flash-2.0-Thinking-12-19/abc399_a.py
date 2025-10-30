import sys

# Read input
N = int(sys.stdin.readline())
S = sys.stdin.readline().strip()
T = sys.stdin.readline().strip()

# Calculate Hamming distance
hamming_distance = 0
for i in range(N):
    if S[i] != T[i]:
        hamming_distance += 1

# Print the result
print(hamming_distance)