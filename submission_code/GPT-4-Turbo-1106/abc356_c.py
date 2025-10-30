from itertools import combinations

# Read the input
N, M, K = map(int, input().split())
tests = [input().split() for _ in range(M)]

# Convert test inputs to integers and separate results
tests = [(list(map(int, test[:-1])), test[-1]) for test in tests]

# Function to check if a set of keys satisfies a test
def satisfies_test(keys_set, test):
    keys, result = test
    count_real_keys = sum(1 for key in keys if key in keys_set)
    if result == 'o':
        return count_real_keys >= K
    else:
        return count_real_keys < K

# Function to count the number of valid combinations
def count_valid_combinations(N, tests):
    valid_combinations = 0
    for keys_set in combinations(range(1, N + 1), K):
        if all(satisfies_test(keys_set, test) for test in tests):
            valid_combinations += 1 << (N - K)  # Multiply by 2^(N-K) for dummy keys
    return valid_combinations

# Calculate and print the number of valid combinations
print(count_valid_combinations(N, tests))