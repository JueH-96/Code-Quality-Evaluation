# YOUR CODE HERE
def min_operations_to_remove_black_cells(t, test_cases):
    results = []
    for case in test_cases:
        n, k, s = case
        min_operations = float('inf')
        
        # Count initial black cells in the first window of size k
        current_black_count = s[:k].count('B')
        min_operations = min(min_operations, current_black_count)
        
        # Sliding window to count black cells in each window of size k
        for i in range(1, n - k + 1):
            if s[i - 1] == 'B':
                current_black_count -= 1
            if s[i + k - 1] == 'B':
                current_black_count += 1
            min_operations = min(min_operations, current_black_count)
        
        results.append(min_operations)
    
    return results

# Reading input
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

# Processing and outputting results
results = min_operations_to_remove_black_cells(t, test_cases)
for result in results:
    print(result)