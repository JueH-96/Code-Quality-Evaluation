def count_valid_key_combinations(N, M, K, tests):
    valid_count = 0
    
    # Iterate over all possible combinations of real and dummy keys
    for mask in range(1 << N):
        real_keys = [i + 1 for i in range(N) if (mask & (1 << i)) != 0]
        real_count = len(real_keys)
        
        # Check if this combination satisfies all tests
        valid = True
        for C, keys, result in tests:
            count_real_in_test = sum(1 for key in keys if (mask & (1 << (key - 1))) != 0)
            
            if result == 'o' and count_real_in_test < K:
                valid = False
                break
            if result == 'x' and count_real_in_test >= K:
                valid = False
                break
        
        if valid:
            valid_count += 1
    
    return valid_count

import sys
input = sys.stdin.read
data = input().splitlines()

# Read N, M, K
N, M, K = map(int, data[0].split())

# Read tests
tests = []
for i in range(1, M + 1):
    parts = data[i].split()
    C = int(parts[0])
    keys = list(map(int, parts[1:C + 1]))
    result = parts[C + 1]
    tests.append((C, keys, result))

# Get the count of valid combinations
result = count_valid_key_combinations(N, M, K, tests)

# Print the result
print(result)