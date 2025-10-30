import sys
import math

def main():
    import sys
    sys.setrecursionlimit(1 << 25)
    N, M = map(int, input().split())
    info = [[] for _ in range(N+1)]
    for _ in range(M):
        l, d, k, c, A, B = map(int, sys.stdin.readline().split())
        info[A].append((l, d, k, c, B))
    
    dp = [float('-inf')] * (N + 1)
    dp[N] = float('inf')
    
    for S in range(N-1, 0, -1):
        for l, d, k, c, B in info[S]:
            if dp[B] == float('-inf'):
                continue
            num = dp[B] - l - c
            if num < 0:
                continue
            if d == 0:
                if num >= 0:
                    arrival_time = l + c
                    dp[S] = max(dp[S], arrival_time)
                continue
            i_max = num // d
            if d > 0:
                if i_max < 0:
                    continue
                if i_max >= k:
                    i_max = k - 1
                arrival_time = l + i_max * d + c
                dp[S] = max(dp[S], arrival_time)
            else:
                # d < 0, which shouldn't happen as per constraints
                pass
    
    for S in range(1, N):
        if dp[S] > float('-inf'):
            print(dp[S])
        else:
            print("Unreachable")

if __name__ == '__main__':
    main()