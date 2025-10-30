def main():
    import sys
    input = sys.stdin.read
    data = input().split()
    
    N = int(data[0])
    X = []
    Y = []
    for i in range(N):
        X.append(int(data[1 + 2*i]))
        Y.append(int(data[2 + 2*i]))
    
    # Initialize DP table
    # dp[i][state] where state is 0 (healthy) or 1 (upset)
    # Initialize with -infinity to represent unreachable states
    INF = -10**18
    dp = [[INF] * 2 for _ in range(N+1)]
    dp[0][0] = 0  # Start with healthy stomach
    
    for i in range(N):
        x = X[i]
        y = Y[i]
        for state in range(2):
            if dp[i][state] == INF:
                continue
            # Option 1: Skip the course
            dp[i+1][state] = max(dp[i+1][state], dp[i][state])
            # Option 2: Eat the course
            if state == 0:
                if x == 0:
                    new_state = 0
                else:
                    new_state = 1
            else:
                if x == 0:
                    new_state = 0
                else:
                    # Eating a poisonous course with upset stomach leads to death
                    continue
            dp[i+1][new_state] = max(dp[i+1][new_state], dp[i][state] + y)
    
    # The final state must be either healthy or upset, but not dead
    result = max(dp[N][0], dp[N][1])
    print(max(result, 0))

if __name__ == "__main__":
    main()