# YOUR CODE HERE
from itertools import combinations

def count_valid_combinations(N, M, K, tests):
    valid_count = 0
    
    # Iterate over all possible combinations of real keys
    for i in range(1 << N):
        real_keys = set()
        for j in range(N):
            if i & (1 << j):
                real_keys.add(j + 1)
        
        # Check if this combination of real keys satisfies all tests
        valid = True
        for test in tests:
            C, keys, result = test
            real_count = sum(1 for key in keys if key in real_keys)
            if (result == 'o' and real_count < K) or (result == 'x' and real_count >= K):
                valid = False
                break
        
        if valid:
            valid_count += 1
    
    return valid_count

# Read input
import sys
input = sys.stdin.read
data = input().split()

N = int(data[0])
M = int(data[1])
K = int(data[2])

tests = []
index = 3
for _ in range(M):
    C = int(data[index])
    keys = list(map(int, data[index + 1:index + 1 + C]))
    result = data[index + 1 + C]
    tests.append((C, keys, result))
    index += 2 + C

# Calculate and print the result
print(count_valid_combinations(N, M, K, tests))