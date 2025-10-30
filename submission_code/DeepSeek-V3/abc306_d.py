# YOUR CODE HERE
def main():
    import sys
    input = sys.stdin.read
    data = input().split()
    
    N = int(data[0])
    X = []
    Y = []
    for i in range(N):
        X.append(int(data[2*i+1]))
        Y.append(int(data[2*i+2]))
    
    # Initialize DP table
    # dp[i][state] where state is 0 (healthy) or 1 (upset)
    # Initialize with -infinity
    dp = [[-float('inf')] * 2 for _ in range(N+1)]
    dp[0][0] = 0  # Start with healthy stomach
    
    for i in range(N):
        x = X[i]
        y = Y[i]
        for state in range(2):
            if dp[i][state] == -float('inf'):
                continue
            # Option 1: Skip the course
            dp[i+1][state] = max(dp[i+1][state], dp[i][state])
            # Option 2: Eat the course
            if x == 0:
                new_state = state ^ 1 if state == 1 else state
            else:
                if state == 1:
                    # Eating a poisonous course with upset stomach leads to death
                    continue
                else:
                    new_state = 1
            dp[i+1][new_state] = max(dp[i+1][new_state], dp[i][state] + y)
    
    # The final state must be either healthy or upset, but not dead
    # So we take the maximum of dp[N][0] and dp[N][1]
    result = max(dp[N][0], dp[N][1])
    print(result)

if __name__ == "__main__":
    main()