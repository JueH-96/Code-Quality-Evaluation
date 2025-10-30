# YOUR CODE HERE
from itertools import combinations

def count_valid_combinations(N, M, K, tests):
    count = 0
    for real_keys in range(1 << N):
        valid = True
        for test in tests:
            C = test[0]
            keys = test[1:-1]
            result = test[-1]
            real_count = sum((real_keys >> (key - 1)) & 1 for key in keys)
            if (real_count >= K and result == 'x') or (real_count < K and result == 'o'):
                valid = False
                break
        if valid:
            count += 1
    return count

N, M, K = map(int, input().split())
tests = []
for _ in range(M):
    test = list(map(int, input().split()))
    test.append(input().split()[-1])
    tests.append(test)

print(count_valid_combinations(N, M, K, tests))