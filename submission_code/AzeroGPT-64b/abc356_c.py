from itertools import product

N, M, K = map(int, input().split())
keys = [set(map(int, input().split(' '))[:-1]) for _ in range(M)]
tests = [input().split(' ')[-1] for _ in range(M)]

def is_valid(keys, tests):
    real_keys = [k for k, b in enumerate(keys, 1) if b]
    for test_keys, test_result in zip(keys, tests):
        real_test_keys = sum([1 for k in test_keys if k in real_keys])
        if (test_result == 'o' and real_test_keys < K) or (test_result == 'x' and real_test_keys >= K):
            return False
    return True

valid_keys = 0
for keys in product([0, 1], repeat=N):
    if is_valid(keys, tests):
        valid_keys += 1

print(valid_keys)