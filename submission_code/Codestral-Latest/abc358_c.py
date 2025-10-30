import sys
from itertools import combinations

# Read input
input = sys.stdin.read
data = input().split()

N = int(data[0])
M = int(data[1])
stands = data[2:]

# Function to check if a combination of stands covers all flavors
def covers_all_flavors(comb):
    covered = [False] * M
    for stand in comb:
        for j in range(M):
            if stand[j] == 'o':
                covered[j] = True
    return all(covered)

# Find the minimum number of stands needed
min_stands = N
for r in range(1, N + 1):
    for comb in combinations(stands, r):
        if covers_all_flavors(comb):
            min_stands = min(min_stands, r)
            break

# Print the result
print(min_stands)