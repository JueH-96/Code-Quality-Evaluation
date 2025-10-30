import sys
from itertools import combinations

# Read input
input = sys.stdin.read
data = input().split()

N = int(data[0])
M = int(data[1])
K = int(data[2])

tests = []
index = 3
for _ in range(M):
    C = int(data[index])
    keys = list(map(int, data[index + 1:index + C + 1]))
    result = data[index + C + 1]
    tests.append((C, keys, result))
    index += C + 2

# Function to check if a combination is valid
def is_valid(combination):
    for C, keys, result in tests:
        real_count = sum(1 for key in keys if combination[key - 1])
        if (result == 'o' and real_count < K) or (result == 'x' and real_count >= K):
            return False
    return True

# Count valid combinations
valid_count = 0
for combination in combinations(range(N), N):
    if is_valid(combination):
        valid_count += 1

# Output the result
print(valid_count)