# YOUR CODE HERE
def count_abc(s):
    count = 0
    for i in range(len(s) - 2):
        if s[i:i+3] == "ABC":
            count += 1
    return count

def process_queries(n, q, s, queries):
    # Initial count of "ABC"
    abc_count = count_abc(s)
    results = []
    
    s = list(s)  # Convert string to list for mutability
    
    for x, c in queries:
        x -= 1  # Convert to 0-based index
        
        # Check the positions around x that could be affected
        for i in range(max(0, x-2), min(n-2, x+1)):
            if s[i:i+3] == ['A', 'B', 'C']:
                abc_count -= 1
        
        # Update the character
        s[x] = c
        
        # Check again the positions around x
        for i in range(max(0, x-2), min(n-2, x+1)):
            if s[i:i+3] == ['A', 'B', 'C']:
                abc_count += 1
        
        # Store the result for this query
        results.append(abc_count)
    
    return results

import sys
input = sys.stdin.read
data = input().split()

n = int(data[0])
q = int(data[1])
s = data[2]

queries = []
index = 3
for _ in range(q):
    x = int(data[index])
    c = data[index + 1]
    queries.append((x, c))
    index += 2

results = process_queries(n, q, s, queries)

for result in results:
    print(result)