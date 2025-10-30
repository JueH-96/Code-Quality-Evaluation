def main():
    import sys
    input = sys.stdin.read
    data = input().split()
    
    MOD = 998244353
    
    N = int(data[0])
    A = list(map(int, data[1:]))
    
    if N == 1:
        print(1)
        return
    
    # Initial state of the grid based on i % 2
    initial = [(i % 2) for i in range(1, N + 1)]
    
    # We need to transform `initial` to `A` using the allowed operations.
    
    # dp[i] will store the number of ways to achieve A[0:i] (1-based index)
    dp = [0] * (N + 1)
    dp[0] = 1  # Base case: 1 way to achieve an empty prefix
    
    # To optimize, we keep last positions where 0 and 1 were seen
    last_seen = [-1, -1]
    
    for i in range(1, N + 1):
        current_value = A[i - 1]
        last_position = last_seen[current_value]
        
        if last_position != -1:
            dp[i] = dp[last_position]
        
        dp[i] += dp[i - 1]
        dp[i] %= MOD
        
        last_seen[current_value] = i
    
    # The answer is the number of ways to achieve the full array A
    print(dp[N])

if __name__ == "__main__":
    main()