def solve():
    import sys
    input = sys.stdin.read
    data = input().split()
    
    MOD = 998244353
    
    N = int(data[0])
    A = list(map(int, data[1:N+1]))
    B = list(map(int, data[N+1:2*N+1]))
    
    # Positions in A and B are 1-based, but we will use 0-based indexing for easier manipulation
    A = [a - 1 for a in A]
    B = [b - 1 if b != -1 else -1 for b in B]
    
    # All possible positions from 0 to 2N-1
    all_positions = set(range(2 * N))
    
    # Fixed positions from A and B
    fixed_positions = set(A)
    for b in B:
        if b != -1:
            fixed_positions.add(b)
    
    # Free positions that are not predetermined by A or B
    free_positions = list(all_positions - fixed_positions)
    
    # Count of free positions
    free_count = len(free_positions)
    
    # We need to assign these free positions to the remaining B[i] = -1
    # and ensure no sequence is the same as its reverse.
    
    # Dynamic programming table
    # dp[i][j] means the number of ways to assign the first i free positions
    # to create valid sequences, where j pairs (s_i, t_i) are already fixed
    # such that s_i = t_i (which should be zero for a valid solution).
    dp = [[0] * (N + 1) for _ in range(free_count + 1)]
    dp[0][0] = 1
    
    # Iterate over each free position
    for i in range(free_count):
        for j in range(N + 1):
            if dp[i][j] == 0:
                continue
            
            # Case 1: Assign this position to a B[k] = -1 where s_k != t_k
            # This increases the number of valid assignments without increasing j
            dp[i + 1][j] += dp[i][j] * (free_count - i - j)
            dp[i + 1][j] %= MOD
            
            # Case 2: Assign this position to create a new s_k = t_k
            # This increases j by 1
            if j + 1 <= N:
                dp[i + 1][j + 1] += dp[i][j]
                dp[i + 1][j + 1] %= MOD
    
    # The answer is the number of ways to assign all free positions such that no s_i = t_i
    print(dp[free_count][0])