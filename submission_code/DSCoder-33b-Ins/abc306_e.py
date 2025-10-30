from sortedcontainers import SortedList

def solve():
    N, K, Q = map(int, input().split())
    A = SortedList([0]*N)
    for _ in range(Q):
        X, Y = map(int, input().split())
        X -= 1
        if A[-K] < Y:
            A.remove(A[-K])
            A.add(Y)
        print(sum(A[-K:]))

solve()