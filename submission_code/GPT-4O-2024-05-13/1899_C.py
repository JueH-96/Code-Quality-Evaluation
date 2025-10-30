# YOUR CODE HERE
def max_alternating_subarray_sum(t, test_cases):
    results = []
    for i in range(t):
        n = test_cases[i][0]
        a = test_cases[i][1]
        
        max_sum = float('-inf')
        current_sum = 0
        last_parity = None
        
        for num in a:
            current_parity = num % 2
            if last_parity is None or current_parity != last_parity:
                current_sum += num
                last_parity = current_parity
            else:
                max_sum = max(max_sum, current_sum)
                current_sum = num
                last_parity = current_parity
        
        max_sum = max(max_sum, current_sum)
        results.append(max_sum)
    
    return results

import sys
input = sys.stdin.read
data = input().split()

t = int(data[0])
index = 1
test_cases = []

for _ in range(t):
    n = int(data[index])
    array = list(map(int, data[index + 1: index + 1 + n]))
    test_cases.append((n, array))
    index += 1 + n

results = max_alternating_subarray_sum(t, test_cases)
for result in results:
    print(result)