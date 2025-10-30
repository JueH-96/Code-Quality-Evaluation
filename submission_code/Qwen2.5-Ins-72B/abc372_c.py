# YOUR CODE HERE
import sys

def count_abc(s):
    count = 0
    for i in range(len(s) - 2):
        if s[i] == 'A' and s[i + 1] == 'B' and s[i + 2] == 'C':
            count += 1
    return count

def process_queries(n, q, s, queries):
    for x, c in queries:
        x -= 1  # Convert to 0-based index
        original_char = s[x]
        s = s[:x] + c + s[x + 1:]
        
        # Calculate the change in the number of "ABC" substrings
        if original_char == 'A' and c != 'A' and x + 2 < n and s[x + 1] == 'B' and s[x + 2] == 'C':
            count -= 1
        elif original_char != 'A' and c == 'A' and x + 2 < n and s[x + 1] == 'B' and s[x + 2] == 'C':
            count += 1
        elif original_char == 'B' and c != 'B' and x > 0 and s[x - 1] == 'A' and x + 1 < n and s[x + 1] == 'C':
            count -= 1
        elif original_char != 'B' and c == 'B' and x > 0 and s[x - 1] == 'A' and x + 1 < n and s[x + 1] == 'C':
            count += 1
        elif original_char == 'C' and c != 'C' and x > 1 and s[x - 2] == 'A' and s[x - 1] == 'B':
            count -= 1
        elif original_char != 'C' and c == 'C' and x > 1 and s[x - 2] == 'A' and s[x - 1] == 'B':
            count += 1
        
        print(count)

# Read input
input = sys.stdin.read
data = input().split()
n = int(data[0])
q = int(data[1])
s = data[2]
queries = [(int(data[3 * i + 3]), data[3 * i + 4]) for i in range(q)]

# Initial count of "ABC" substrings
count = count_abc(s)

# Process queries
process_queries(n, q, s, queries)