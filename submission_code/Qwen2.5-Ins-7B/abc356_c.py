# YOUR CODE HERE
from itertools import combinations

def is_valid_combination(combination, tests):
    for C, keys, result in tests:
        real_keys = sum(1 for key in keys if key in combination)
        if (result == 'o' and real_keys < K) or (result == 'x' and real_keys >= K):
            return False
    return True

N, M, K = map(int, input().split())
tests = []
for _ in range(M):
    C, *keys, result = input().split()
    tests.append((int(C), keys, result))

valid_combinations = 0
for combination in combinations(range(1, N+1), K):
    if is_valid_combination(combination, tests):
        valid_combinations += 1

print(valid_combinations)