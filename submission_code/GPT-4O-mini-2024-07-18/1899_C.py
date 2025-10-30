def max_alternating_subarray_sum(t, test_cases):
    results = []
    for case in test_cases:
        n, a = case
        max_sum = float('-inf')
        current_sum = 0
        
        for i in range(n):
            if i == 0 or (a[i] % 2) != (a[i - 1] % 2):
                current_sum += a[i]
            else:
                current_sum = a[i]
            
            max_sum = max(max_sum, current_sum)
        
        results.append(max_sum)
    
    return results

# Read input
import sys
input = sys.stdin.read
data = input().splitlines()

t = int(data[0])
test_cases = []
index = 1

for _ in range(t):
    n = int(data[index])
    a = list(map(int, data[index + 1].split()))
    test_cases.append((n, a))
    index += 2

# Get results
results = max_alternating_subarray_sum(t, test_cases)

# Print results
for result in results:
    print(result)