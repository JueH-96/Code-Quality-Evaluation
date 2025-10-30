# YOUR CODE HERE
def min_operations_to_remove_black_cells(test_cases):
    results = []
    for n, k, s in test_cases:
        # Initial count of 'B's in the first window of size k
        current_black_count = sum(1 for i in range(k) if s[i] == 'B')
        min_operations = current_black_count
        
        # Slide the window across the string
        for i in range(1, n - k + 1):
            # Remove the effect of the character sliding out of the window
            if s[i - 1] == 'B':
                current_black_count -= 1
            # Add the effect of the character sliding into the window
            if s[i + k - 1] == 'B':
                current_black_count += 1
            # Update the minimum operations
            min_operations = min(min_operations, current_black_count)
        
        results.append(min_operations)
    
    return results

# Reading input
import sys
input = sys.stdin.read
data = input().split()

# Parsing input
t = int(data[0])
index = 1
test_cases = []
for _ in range(t):
    n = int(data[index])
    k = int(data[index + 1])
    s = data[index + 2]
    test_cases.append((n, k, s))
    index += 3

# Solving the problem
results = min_operations_to_remove_black_cells(test_cases)

# Printing the results
for result in results:
    print(result)