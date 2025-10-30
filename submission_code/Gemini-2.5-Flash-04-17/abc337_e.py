import sys

# Read N
N = int(sys.stdin.readline())

# Calculate the minimum number of friends M
# M is the smallest integer such that 2^M >= N
# This is equivalent to M = ceil(log2(N))
# We can find M iteratively:
M = 0
current_pow_of_2 = 1
while current_pow_of_2 < N:
    current_pow_of_2 *= 2
    M += 1

# Print M
print(M)
sys.stdout.flush()

# Process and print the bottles for each friend (Friend 1 to M)
# Friend i (1-indexed) corresponds to bit i-1 (0-indexed)
# Friend i receives bottle j (1-indexed) if bit i-1 of (j-1) is 1.
for i in range(M): # Iterate through friend indices 0 to M-1 (corresponding to bits 0 to M-1)
    bottles_for_current_friend = []
    # Iterate through bottles 1 to N
    for j in range(1, N + 1):
        bottle_index_minus_one = j - 1
        # Check if the i-th bit (0-indexed) of bottle_index_minus_one is 1
        # The i-th bit corresponds to friend i+1
        if (bottle_index_minus_one >> i) & 1:
            bottles_for_current_friend.append(j)

    # Print the distribution for the current friend (friend i+1)
    # Print the number of bottles followed by the bottle numbers in ascending order
    print(len(bottles_for_current_friend), *bottles_for_current_friend)
    sys.stdout.flush() # Ensure output is sent after each line

# Read the outcome string S
# S has length M, S[i] is the outcome for friend i+1
S = sys.stdin.readline().strip()

# Interpret S as a binary number (LSB first)
# S[0] is the outcome for friend 1, corresponding to bit 0 of X-1
# S[1] is the outcome for friend 2, corresponding to bit 1 of X-1
# ...
# S[M-1] is the outcome for friend M, corresponding to bit M-1 of X-1
# The value v = sum(S[i] * 2^i) where S[i] is '0' or '1', corresponds to X-1
spoiled_bottle_index_minus_one = 0
for i in range(M):
    # If friend i+1 got sick (S[i] is '1'), it means bit i of (X-1) is 1
    if S[i] == '1':
        spoiled_bottle_index_minus_one += (1 << i)

# The spoiled bottle number is (decoded binary value) + 1
spoiled_bottle_number = spoiled_bottle_index_minus_one + 1

# Print the spoiled bottle number
print(spoiled_bottle_number)
# No need to flush after the final output as the program terminates.