def main():
    import sys
    K = int(sys.stdin.readline())
    S = sys.stdin.readline().strip()
    T = sys.stdin.readline().strip()
    
    N = len(S)
    M = len(T)
    
    if N > M:
        S, T = T, S
        N, M = M, N
    
    if N + K < M:
        print("No")
        return
    
    dp_prev = list(range(M + 1))
    
    for i in range(1, N + 1):
        dp = [0] * (M + 1)
        dp[0] = i
        for j in range(max(1, i - K), min(M, i + K) + 1):
            if S[i - 1] == T[j - 1]:
                dp[j] = dp_prev[j - 1]
            else:
                dp[j] = 1 + min(dp_prev[j - 1], dp_prev[j], dp[j - 1])
        dp_prev = dp
    
    if dp[M] <= K:
        print("Yes")
    else:
        print("No")

if __name__ == "__main__":
    main()