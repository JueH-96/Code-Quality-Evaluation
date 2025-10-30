def min_operations_to_remove_black_cells(t, test_cases):
    results = []
    for n, k, s in test_cases:
        operations = 0
        i = 0
        
        while i < n:
            if s[i] == 'B':
                operations += 1
                # Move i forward by k to simulate the operation
                i += k
            else:
                i += 1
        
        results.append(operations)
    
    return results

# Read input
import sys
input = sys.stdin.read
data = input().splitlines()

t = int(data[0])
test_cases = []

for i in range(t):
    n, k = map(int, data[2 * i + 1].split())
    s = data[2 * i + 2]
    test_cases.append((n, k, s))

# Get results
results = min_operations_to_remove_black_cells(t, test_cases)

# Print results
for result in results:
    print(result)