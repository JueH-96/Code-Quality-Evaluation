import math

# Read N
N = int(input())

# Calculate minimum number of friends needed
M = math.ceil(math.log2(N)) if N > 1 else 1

# Print number of friends
print(M)

# For each friend, give bottles based on binary representation
for i in range(M):
    bottles = []
    for bottle in range(1, N + 1):
        # Check if bit i is set in bottle number (bottle-1 since we start from 1)
        if (bottle - 1) & (1 << i):
            bottles.append(bottle)
    
    # Print the bottles for this friend
    if bottles:
        print(len(bottles), *bottles)
    else:
        print(0)

# Read the result string
S = input().strip()

# Decode the spoiled bottle number from binary
spoiled_bottle = 0
for i in range(M):
    if S[i] == '1':
        spoiled_bottle |= (1 << i)

# Convert back to 1-indexed
spoiled_bottle += 1

# Print the answer
print(spoiled_bottle)