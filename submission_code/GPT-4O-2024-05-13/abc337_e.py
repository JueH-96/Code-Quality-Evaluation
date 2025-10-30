# YOUR CODE HERE
import sys
import math

input = sys.stdin.read
data = input().strip().split()
N = int(data[0])

# Calculate the number of friends needed
M = math.ceil(math.log2(N))

# Print the number of friends
print(M)
sys.stdout.flush()

# Distribute the bottles to friends using binary representation
for i in range(M):
    bottles = []
    for j in range(1, N + 1):
        if (j >> i) & 1:
            bottles.append(j)
    print(len(bottles), ' '.join(map(str, bottles)))
    sys.stdout.flush()

# Read the results from the judge
S = input().strip()

# Determine the spoiled bottle
spoiled_bottle = 0
for i in range(M):
    if S[i] == '1':
        spoiled_bottle |= (1 << i)

# Print the spoiled bottle number
print(spoiled_bottle)
sys.stdout.flush()