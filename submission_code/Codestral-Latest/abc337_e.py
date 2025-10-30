import sys
import math

# Read the number of bottles
N = int(input().strip())

# Determine the number of friends to call
M = math.ceil(math.log2(N))

# Print the number of friends to call
print(M)

# Prepare the distribution of bottles
distributions = []
for i in range(M):
    bottles = [j for j in range(1, N+1) if (j >> i) & 1]
    print(len(bottles), *bottles)
    sys.stdout.flush()

# Read the response from the judge
response = input().strip()

# Determine the spoiled bottle
spoiled_bottle = 0
for i in range(M):
    if response[i] == '1':
        spoiled_bottle += 1 << i

# Print the spoiled bottle number
print(spoiled_bottle)