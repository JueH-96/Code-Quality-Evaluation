# YOUR CODE HERE
def process_queries(queries):
    A = []
    results = []
    
    for query in queries:
        parts = query.split()
        if parts[0] == '1':
            # Append x to the end of A
            x = int(parts[1])
            A.append(x)
        elif parts[0] == '2':
            # Find the k-th value from the end of A
            k = int(parts[1])
            results.append(A[-k])
    
    return results

import sys

input = sys.stdin.read
data = input().strip().splitlines()
Q = int(data[0])
queries = data[1:Q + 1]

results = process_queries(queries)
for result in results:
    print(result)