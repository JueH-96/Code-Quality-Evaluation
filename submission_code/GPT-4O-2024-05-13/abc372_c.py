# YOUR CODE HERE
def count_abc(s):
    count = 0
    for i in range(len(s) - 2):
        if s[i:i+3] == 'ABC':
            count += 1
    return count

import sys
input = sys.stdin.read
data = input().split()

N = int(data[0])
Q = int(data[1])
S = list(data[2])
queries = data[3:]

# Initial count of 'ABC' in the string
abc_count = count_abc(S)

results = []
for i in range(Q):
    X_i = int(queries[2 * i]) - 1
    C_i = queries[2 * i + 1]
    
    # Check the positions that might be affected by the change
    start = max(0, X_i - 2)
    end = min(N - 1, X_i + 2)
    
    # Decrease count for the current 'ABC' substrings in the affected range
    for j in range(start, end - 1):
        if S[j:j+3] == ['A', 'B', 'C']:
            abc_count -= 1
    
    # Apply the change
    S[X_i] = C_i
    
    # Increase count for the new 'ABC' substrings in the affected range
    for j in range(start, end - 1):
        if S[j:j+3] == ['A', 'B', 'C']:
            abc_count += 1
    
    # Store the result for the current query
    results.append(abc_count)

# Print all results
for result in results:
    print(result)