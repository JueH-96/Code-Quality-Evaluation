def solve():
    import sys
    data = sys.stdin.read().strip().split()
    N = int(data[0])
    A = list(map(int, data[1:1+N]))
    ST = list(map(int, data[1+N:]))

    # dp[i] will store the current units of currency i+1 that Takahashi has
    dp = A[:]  # start with the initial amounts

    # We greedily convert as much as possible from currency i to i+1
    idx = 0
    for i in range(N-1):
        S_i = ST[idx]
        T_i = ST[idx+1]
        idx += 2
        
        # Number of times we can convert from currency i to i+1
        times = dp[i] // S_i
        
        # Convert
        dp[i+1] += times * T_i
        dp[i] -= times * S_i

    # Print the final amount of currency N
    print(dp[N-1])

def main():
    solve()

if __name__ == "__main__":
    main()