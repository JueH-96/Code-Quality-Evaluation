def solve():
    N = int(input())
    P = [list(map(int, input().split())) for _ in range(N)]
    Q = [list(map(int, input().split())) for _ in range(N)]
    P.sort(key=lambda x: (x[0], x[1]))
    Q.sort(key=lambda x: (x[0], x[1]))
    R = [0]*N
    used = [False]*N
    for i in range(N):
        for j in range(N):
            if not used[j] and P[i][0] <= Q[j][0] and P[i][1] <= Q[j][1]:
                R[i] = j+1
                used[j] = True
                break
        else:
            print(-1)
            return
    print(*R)

solve()