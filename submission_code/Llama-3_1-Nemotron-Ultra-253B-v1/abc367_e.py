import sys

def main():
    N, K = map(int, sys.stdin.readline().split())
    X = list(map(lambda x: int(x)-1, sys.stdin.readline().split()))
    A = list(map(int, sys.stdin.readline().split()))
    
    if K == 0:
        print(' '.join(map(str, A)))
        return
    
    LOG = 60
    dp = [[0] * N for _ in range(LOG)]
    dp[0] = X[:]
    
    for j in range(1, LOG):
        for i in range(N):
            dp[j][i] = dp[j-1][dp[j-1][i]]
    
    res = [0] * N
    for i in range(N):
        current = i
        for j in range(LOG-1, -1, -1):
            if K & (1 << j):
                current = dp[j][current]
        res[i] = current
    
    print(' '.join(map(str, [A[x] for x in res])))

if __name__ == '__main__':
    main()