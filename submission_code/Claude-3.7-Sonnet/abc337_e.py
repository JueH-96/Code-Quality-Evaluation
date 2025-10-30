import math

# Read number of bottles
N = int(input())

# Calculate minimum number of friends needed (using binary encoding)
M = math.ceil(math.log2(N))

# Output number of friends
print(M)

# For each friend (representing a bit position)
for bit in range(M):
    # Find bottles where the bit-th bit is 1
    bottles = []
    for bottle in range(1, N+1):
        # Adjust bottle number to 0-based for bit operations
        if ((bottle-1) >> bit) & 1:
            bottles.append(bottle)
    
    # Output the bottles this friend will test
    K = len(bottles)
    if K > 0:
        print(K, ' '.join(map(str, bottles)))
    else:
        print(0)

# Read which friends got sick
S = input()

# Determine the bad bottle from sick pattern
bad_bottle_idx = 0
for i in range(M):
    if S[i] == '1':
        bad_bottle_idx |= (1 << i)

# Convert to 1-based bottle number
bad_bottle = bad_bottle_idx + 1

# Output the bad bottle (ensuring it's within valid range)
if bad_bottle <= N:
    print(bad_bottle)
else:
    print(1)  # This should never happen with proper constraints