N = int(input())
A = list(map(int, input().split()))
Q = int(input())
queries = [list(map(int, input().split())) for _ in range(Q)]

for query in queries:
    if query[0] == 1:
        index = A.index(query[1])
        A.insert(index + 1, query[2])
    elif query[0] == 2:
        A.remove(query[1])

print(' '.join(map(str, A)))