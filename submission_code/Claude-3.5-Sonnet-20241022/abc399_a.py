# Read input
N = int(input())
S = input()
T = input()

# Calculate Hamming distance
hamming_distance = 0
for i in range(N):
    if S[i] != T[i]:
        hamming_distance += 1

# Output result
print(hamming_distance)