def max_absolute_difference(t, test_cases):
    results = []
    for n, weights in test_cases:
        # Sort the weights to easily find the max and min sums for any k
        weights.sort()
        
        # The maximum sum will be the sum of the last k elements
        # The minimum sum will be the sum of the first k elements
        max_sum = sum(weights[-(n // 2):])  # Sum of the heaviest half
        min_sum = sum(weights[:(n // 2)])   # Sum of the lightest half
        
        # The maximum absolute difference
        max_difference = max_sum - min_sum
        results.append(max_difference)
    
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
    weights = list(map(int, data[index + 1].split()))
    test_cases.append((n, weights))
    index += 2

# Get results
results = max_absolute_difference(t, test_cases)

# Print results
for result in results:
    print(result)