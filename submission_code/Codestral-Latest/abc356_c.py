from itertools import combinations

# Read input
N, M, K = map(int, input().split())
tests = []
for _ in range(M):
    test = input().split()
    C_i = int(test[0])
    A_i = list(map(int, test[1:C_i + 1]))
    R_i = test[-1]
    tests.append((A_i, R_i))

# Function to check if a combination of real keys satisfies all tests
def is_valid_combination(real_keys):
    for A_i, R_i in tests:
        real_keys_in_test = [key for key in A_i if key in real_keys]
        if R_i == 'o' and len(real_keys_in_test) < K:
            return False
        if R_i == 'x' and len(real_keys_in_test) >= K:
            return False
    return True

# Count valid combinations
valid_combinations = 0
for r in range(K, N + 1):
    for real_keys in combinations(range(1, N + 1), r):
        if is_valid_combination(real_keys):
            valid_combinations += 1

print(valid_combinations)