# YOUR CODE HERE
import math

# Read N
N = int(input())

# Calculate minimum number of friends needed
M = math.ceil(math.log2(N))
print(M)

# For each friend, determine which bottles they should drink
for bit in range(M):
    bottles = []
    for bottle in range(1, N + 1):
        # Check if bit 'bit' is set in (bottle - 1)
        if (bottle - 1) & (1 << bit):
            bottles.append(bottle)
    
    # Print the bottles for this friend
    print(len(bottles), *bottles)

# Read which friends got sick
S = input().strip()

# Decode the spoiled bottle number
spoiled = 0
for i in range(M):
    if S[i] == '1':
        spoiled |= (1 << i)

# Add 1 because bottles are 1-indexed
spoiled += 1

# Print the answer
print(spoiled)