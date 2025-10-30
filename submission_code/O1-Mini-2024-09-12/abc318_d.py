# YOUR CODE HERE
import sys

def main():
    import sys
    from sys import stdin
    N_and_rest = sys.stdin.read().split()
    N = int(N_and_rest[0])
    data = list(map(int, N_and_rest[1:]))
    D = [[0]*N for _ in range(N)]
    idx = 0
    for i in range(N):
        for j in range(i+1, N):
            D[i][j] = data[idx]
            D[j][i] = D[i][j]
            idx +=1
    size = 1<<N
    dp = [-1]*(size)
    dp[0]=0
    for mask in range(size):
        if dp[mask]==-1:
            continue
        # Find first unmatched vertex
        try:
            u = next(i for i in range(N) if not (mask & (1<<i)))
        except StopIteration:
            continue
        for v in range(u+1, N):
            if not (mask & (1<<v)):
                new_mask = mask | (1<<u) | (1<<v)
                if dp[new_mask] < dp[mask] + D[u][v]:
                    dp[new_mask] = dp[mask] + D[u][v]
    print(max(dp))

if __name__ == "__main__":
    main()