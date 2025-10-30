import sys

def solve():
    N, M = map(int, input().split())
    operations = []
    for _ in range(M):
        L, R = map(int, input().split())
        operations.append((L, R))

    # Check if the goal is achievable
    if all(L == 1 and R == N for L, R in operations):
        print(1)
        print(1, *[1] * M)
        return

    # Compute the minimum cost to achieve the goal
    cost = 0
    ops = []
    for L, R in operations:
        if all(L <= i <= R for i in range(1, N+1)):
            ops.append(1)
            cost += 1
        elif all(L > i or i > R for i in range(1, N+1)):
            ops.append(2)
            cost += 1
        else:
            ops.append(0)

    if all(x == 1 for x in range(1, N+1)):
        print(cost)
        print(*ops)
    else:
        print(-1)

solve()