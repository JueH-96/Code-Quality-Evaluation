# Read the integer N
N = int(input())

# Read the strings S and T
S = input()
T = input()

# Initialize the Hamming distance counter
hamming_distance = 0

# Iterate through the strings character by character
# Since both strings have length N, we can iterate from index 0 to N-1
for i in range(N):
    # Compare the characters at the current index
    if S[i] != T[i]:
        # If the characters are different, increment the counter
        hamming_distance += 1

# Print the calculated Hamming distance
print(hamming_distance)