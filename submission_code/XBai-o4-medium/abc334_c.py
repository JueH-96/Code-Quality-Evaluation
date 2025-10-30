def main():
    import sys
    input = sys.stdin.read().split()
    N = int(input[0])
    K = int(input[1])
    A = list(map(int, input[2:2+K]))
    
    if K == 0:
        print(0)
        return
    
    dp = [0] * (K + 1)
    dp[0] = 0
    dp[1] = 0
    
    for i in range(2, K + 1):
        if i % 2 == 0:
            dp[i] = dp[i-2] + A[i-1] - A[i-2]
        else:
            dp[i] = min(dp[i-1], dp[i-2] + A[i-1] - A[i-2])
    
    print(dp[K])
    
if __name__ == "__main__":
    main()