from collections import deque

def solve():
    N = int(input())
    A = list(map(int, input().split()))
    Q = int(input())
    queries = [list(map(int, input().split())) for _ in range(Q)]

    index = {a: i for i, a in enumerate(A)}
    A = deque(A)

    for query in queries:
        if query[0] == 1:
            x, y = query[1], query[2]
            i = index[x]
            A.insert(i+1, y)
            for k in range(i+2, len(A)):
                index[A[k]] = k
            index[y] = i+1
        else:
            x = query[1]
            i = index[x]
            del index[x]
            A.remove(x)
            for k in range(i, len(A)):
                index[A[k]] = k

    print(' '.join(map(str, A)))

solve()