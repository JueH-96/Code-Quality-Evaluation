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
    query_type = int(data[index])
    if query_type == 1 or query_type == 2:
        i = int(data[index + 1]) - 1
        x = int(data[index + 2])
        queries.append((query_type, i, x))
        index += 3
    else:
        l = int(data[index + 1]) - 1
        r = int(data[index + 2]) - 1
        queries.append((query_type, l, r))
        index += 3

results = []
for query in queries:
    if query[0] == 1:
        A[query[1]] = query[2]
    elif query[0] == 2:
        B[query[1]] = query[2]
    else:
        l, r = query[1], query[2]
        max_v = 0
        for i in range(l, r + 1):
            max_v = max(max_v + A[i], max_v * B[i])
        results.append(max_v)

for result in results:
    print(result)