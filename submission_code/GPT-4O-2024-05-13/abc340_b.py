# YOUR CODE HERE
import sys
input = sys.stdin.read

data = input().split()
Q = int(data[0])
queries = data[1:]

A = []
output = []

i = 0
while i < len(queries):
    if queries[i] == '1':
        x = int(queries[i + 1])
        A.append(x)
        i += 2
    elif queries[i] == '2':
        k = int(queries[i + 1])
        output.append(A[-k])
        i += 2

for result in output:
    print(result)