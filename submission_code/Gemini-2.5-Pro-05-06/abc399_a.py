# Read the integer N, which is the length of the strings.
N = int(input())

# Read the string S.
S = input()

# Read the string T.
T = input()

# Initialize a counter for the Hamming distance.
hamming_distance = 0

# Iterate through the characters of the strings from index 0 to N-1.
# The problem statement uses 1-based indexing (1 <= i <= N),
# which corresponds to 0-based indexing (0 <= i <= N-1) in Python strings.
for i in range(N):
    # If the characters at the current position i are different,
    # increment the Hamming distance.
    if S[i] != T[i]:
        hamming_distance += 1

# Print the final Hamming distance.
print(hamming_distance)