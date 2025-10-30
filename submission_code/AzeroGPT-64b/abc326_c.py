def main():
    N, M = map(int, input().split())
    A = list(map(int, input().split()))

    if M >= 1000000000:
        print(N)
        return

    A.sort()
    dp = [0] * (N + 1)
    
    winL, winR = 0, M
    for i in range(N):
        while winR < A[i] or winL < i:
            if winL < i:
                dp[i] = dp[i-1]
                winL += 1
            dp[i] += 1
            winR += 1
        
        if winR <= N:
            dp[i] += dp[winR] - dp[i+1]

    print(max(dp))

if __name__ == '__main__':
    main()