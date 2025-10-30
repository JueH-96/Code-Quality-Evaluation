def max_diff(test_cases):
    results = []
    for case in test_cases:
        n, a = case
        max_diff_val = 0
        # Precompute prefix sums
        prefix = [0] * (n + 1)
        for i in range(n):
            prefix[i+1] = prefix[i] + a[i]
        # Iterate over all possible k
        for k in range(1, n+1):
            if n % k != 0:
                continue
            # Calculate the sum for each truck
            truck_sums = []
            for i in range(0, n, k):
                truck_sum = prefix[i+k] - prefix[i]
                truck_sums.append(truck_sum)
            # Find the max and min sum
            current_max = max(truck_sums)
            current_min = min(truck_sums)
            current_diff = current_max - current_min
            if current_diff > max_diff_val:
                max_diff_val = current_diff
        results.append(max_diff_val)
    return results

# Read input
t = int(input())
test_cases = []
for _ in range(t):
    n = int(input())
    a = list(map(int, input().split()))
    test_cases.append((n, a))

# Compute results
results = max_diff(test_cases)

# Output results
for res in results:
    print(res)