import math
import sys

# Read N
N = int(input())

# Calculate minimum number of friends
M = math.ceil(math.log2(N))
print(M)
sys.stdout.flush()

# For each friend, determine which bottles to give
for i in range(M):
    bottles = []
    for j in range(1, N + 1):
        if (j - 1) & (1 << i):  # Check if i-th bit is 1
            bottles.append(j)
    
    # Print the bottles for this friend
    print(len(bottles), *bottles)
    sys.stdout.flush()

# Read the result
S = input().strip()

# Convert binary string to decimal
spoiled = 0
for i in range(M):
    if S[i] == '1':
        spoiled |= (1 << i)

# The spoiled bottle is spoiled + 1 (1-indexed)
print(spoiled + 1)
sys.stdout.flush()