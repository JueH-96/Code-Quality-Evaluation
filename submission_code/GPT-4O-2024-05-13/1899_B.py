# YOUR CODE HERE
def max_abs_diff(t, test_cases):
    results = []
    for i in range(t):
        n, a = test_cases[i]
        total_sum = sum(a)
        max_weight = max(a)
        min_weight = min(a)
        
        # The maximum absolute difference will be between the heaviest and lightest truck
        # which is max_weight - min_weight
        results.append(max_weight - min_weight)
    
    return results

import sys
input = sys.stdin.read
data = input().split()

t = int(data[0])
index = 1
test_cases = []
for _ in range(t):
    n = int(data[index])
    a = list(map(int, data[index + 1: index + 1 + n]))
    test_cases.append((n, a))
    index += 1 + n

results = max_abs_diff(t, test_cases)
for result in results:
    print(result)