import sys

# Read N
N = int(sys.stdin.readline())

# Read S
S = sys.stdin.readline().strip()

# Read T
T = sys.stdin.readline().strip()

# Initialize hamming distance counter
hamming_distance = 0

# Iterate through the strings from index 0 to N-1
for i in range(N):
    # Compare the characters at the current index
    if S[i] != T[i]:
        # If the characters are different, increment the counter
        hamming_distance += 1

# Print the calculated Hamming distance
print(hamming_distance)