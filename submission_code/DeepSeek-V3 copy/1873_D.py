def min_operations(t, test_cases):
    results = []
    for case in test_cases:
        n, k, s = case
        operations = 0
        i = 0
        while i < n:
            if s[i] == 'B':
                operations += 1
                i += k
            else:
                i += 1
        results.append(operations)
    return results

# Read input
t = int(input())
test_cases = []
for _ in range(t):
    n, k = map(int, input().split())
    s = input().strip()
    test_cases.append((n, k, s))

# Compute results
results = min_operations(t, test_cases)

# Output results
for res in results:
    print(res)