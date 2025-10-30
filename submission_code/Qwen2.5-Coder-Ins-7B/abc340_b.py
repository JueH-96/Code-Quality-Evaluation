# YOUR CODE HERE
import sys

input = sys.stdin.read
data = input().split()

Q = int(data[0])
queries = data[1:]

A = []
results = []

index = 0
for query in queries:
    if query[0] == '1':
        x = int(query[2:])
        A.append(x)
    elif query[0] == '2':
        k = int(query[2:])
        results.append(A[-k])

for result in results:
    print(result)