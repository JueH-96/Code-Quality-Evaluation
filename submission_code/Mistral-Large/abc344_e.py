import sys
input = sys.stdin.read
data = input().split()

N = int(data[0])
A = list(map(int, data[1:N+1]))
Q = int(data[N+1])
queries = data[N+2:]

index_map = {value: idx for idx, value in enumerate(A)}

query_pointer = 0
for _ in range(Q):
    query_type = int(queries[query_pointer])
    x = int(queries[query_pointer + 1])

    if query_type == 1:
        y = int(queries[query_pointer + 2])
        idx = index_map[x]
        A.insert(idx + 1, y)
        for i in range(idx + 1, len(A)):
            index_map[A[i]] = i
        index_map[y] = idx + 1
        query_pointer += 3
    elif query_type == 2:
        idx = index_map[x]
        A.pop(idx)
        for i in range(idx, len(A)):
            index_map[A[i]] = i
        del index_map[x]
        query_pointer += 2

print(" ".join(map(str, A)))