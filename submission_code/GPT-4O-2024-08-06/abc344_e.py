# YOUR CODE HERE
def process_queries(N, A, Q, queries):
    # Convert A to a list for easy manipulation
    A = list(A)
    
    for query in queries:
        parts = query.split()
        if parts[0] == '1':
            # Insert y immediately after x
            x = int(parts[1])
            y = int(parts[2])
            index = A.index(x)
            A.insert(index + 1, y)
        elif parts[0] == '2':
            # Remove x
            x = int(parts[1])
            A.remove(x)
    
    return A

import sys
input = sys.stdin.read
data = input().splitlines()

# Read N
N = int(data[0])

# Read A
A = tuple(map(int, data[1].split()))

# Read Q
Q = int(data[2])

# Read Queries
queries = data[3:3 + Q]

# Process queries
result = process_queries(N, A, Q, queries)

# Print the result
print(' '.join(map(str, result)))