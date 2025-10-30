import sys
import threading
def main():
    import sys
    sys.setrecursionlimit(1 << 25)
    mod = 998244353
    data = sys.stdin.read().split()
    if not data:
        return
    it = iter(data)
    N = int(next(it))
    # We work on a grid from x=0..N+1, y=0..N+1
    # Ball positions are at (x,y) for x,y in [1..N]
    has = [[False] * (N+2) for _ in range(N+2)]
    for _ in range(N):
        x = int(next(it)); y = int(next(it))
        has[x][y] = True
    # dp[x][y][0 or 1]: number of ways to reach (x,y) with last move dir:
    # 0 = last was E (right), 1 = last was S (down)
    dp0 = [[0] * (N+2) for _ in range(N+2)]
    dp1 = [[0] * (N+2) for _ in range(N+2)]
    # Start at (0, N+1), we pretend last move was S so that an immediate E is allowed at start
    dp1[0][N+1] = 1
    for x in range(0, N+2):
        # y goes from high down to 0 so that dp[x][y+1] is already processed when needed
        for y in range(N+1, -1, -1):
            v0 = dp0[x][y]
            if v0:
                # from last move E, we can always go E or S
                if x + 1 <= N+1:
                    dp0[x+1][y] = (dp0[x+1][y] + v0) % mod
                if y - 1 >= 0:
                    dp1[x][y-1] = (dp1[x][y-1] + v0) % mod
            v1 = dp1[x][y]
            if v1:
                # from last move S, we can continue S freely
                if y - 1 >= 0:
                    dp1[x][y-1] = (dp1[x][y-1] + v1) % mod
                # from S to E, we must check the "corner" condition
                if x + 1 <= N+1:
                    # allowed if (x,y) is a ball, or is start or is end
                    if (x == 0 and y == N+1) or (x == N+1 and y == 0) or (1 <= x <= N and 1 <= y <= N and has[x][y]):
                        dp0[x+1][y] = (dp0[x+1][y] + v1) % mod
    # answer is sum of ways to reach (N+1, 0) with either last move
    ans = (dp0[N+1][0] + dp1[N+1][0]) % mod
    print(ans)

if __name__ == "__main__":
    main()