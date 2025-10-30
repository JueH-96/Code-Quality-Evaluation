def min_operations_to_remove_blacks(t, test_cases):
    results = []
    for n, k, s in test_cases:
        # Count the number of black cells
        total_blacks = s.count('B')
        
        if total_blacks == 0:
            results.append(0)
            continue
        
        # We need to find the minimum number of operations to cover all black cells
        # We use a sliding window of size k to find the window with the maximum number of black cells
        # This will minimize the number of operations needed
        
        # Initial count of black cells in the first window of size k
        current_blacks = s[:k].count('B')
        max_blacks_in_window = current_blacks
        
        # Sliding window to find the maximum black cells in any window of size k
        for i in range(1, n - k + 1):
            if s[i - 1] == 'B':
                current_blacks -= 1
            if s[i + k - 1] == 'B':
                current_blacks += 1
            max_blacks_in_window = max(max_blacks_in_window, current_blacks)
        
        # The minimum operations needed is the total black cells minus the maximum black cells we can cover in one operation
        min_operations = total_blacks - max_blacks_in_window
        results.append(min_operations)
    
    return results

import sys
input = sys.stdin.read
data = input().split()

t = int(data[0])
index = 1
test_cases = []

for _ in range(t):
    n = int(data[index])
    k = int(data[index + 1])
    s = data[index + 2]
    test_cases.append((n, k, s))
    index += 3

results = min_operations_to_remove_blacks(t, test_cases)
for result in results:
    print(result)