# YOUR CODE HERE
import sys
input = sys.stdin.read

data = input().split()
index = 0

N = int(data[index])
index += 1

A = list(map(int, data[index:index + N]))
index += N

B = list(map(int, data[index:index + N]))
index += N

Q = int(data[index])
index += 1

queries = []
for _ in range(Q):
    queries.append(data[index:index + 3])
    index += 3

results = []

for query in queries:
    if query[0] == '1':
        i = int(query[1]) - 1
        x = int(query[2])
        A[i] = x
    elif query[0] == '2':
        i = int(query[1]) - 1
        x = int(query[2])
        B[i] = x
    elif query[0] == '3':
        l = int(query[1]) - 1
        r = int(query[2]) - 1
        v = 0
        for i in range(l, r + 1):
            v = max(v + A[i], v * B[i])
        results.append(v)

for result in results:
    print(result)