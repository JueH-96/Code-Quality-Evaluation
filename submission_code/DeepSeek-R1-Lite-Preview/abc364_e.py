def main():
    import sys
    input = sys.stdin.read
    data = input().split()
    
    N = int(data[0])
    X = int(data[1])
    Y = int(data[2])
    A = [int(data[2*i+3]) for i in range(N)]
    B = [int(data[2*i+4]) for i in range(N)]
    
    INF = float('inf')
    dp = [[INF] * (Y + 1) for _ in range(N + 1)]
    dp[0][0] = 0
    
    for idx in range(N):
        a = A[idx]
        b = B[idx]
        for k in range(N, 0, -1):
            for t in range(Y - b, -1, -1):
                if dp[k - 1][t] + a <= X:
                    dp[k][t + b] = min(dp[k][t + b], dp[k - 1][t] + a)
    
    for k in range(N, 0, -1):
        for t in range(Y + 1):
            if dp[k][t] <= X and t <= Y:
                print(k)
                return
    print(0)

if __name__ == "__main__":
    main()