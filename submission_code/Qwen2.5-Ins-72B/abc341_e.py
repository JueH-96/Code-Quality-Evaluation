# YOUR CODE HERE
import sys

def is_good_string(s):
    for i in range(len(s) - 1):
        if s[i] == s[i + 1]:
            return False
    return True

def process_queries(n, q, s, queries):
    results = []
    for query in queries:
        if query[0] == 1:
            l, r = query[1], query[2]
            s = s[:l-1] + s[l-1:r].translate(str.maketrans('01', '10')) + s[r:]
        elif query[0] == 2:
            l, r = query[1], query[2]
            substring = s[l-1:r]
            results.append("Yes" if is_good_string(substring) else "No")
    return results

# Read input
input = sys.stdin.read
data = input().split()
n, q = map(int, data[0].split())
s = data[1]
queries = [list(map(int, data[i].split())) for i in range(2, 2 + q)]

# Process queries and print results
results = process_queries(n, q, s, queries)
for result in results:
    print(result)