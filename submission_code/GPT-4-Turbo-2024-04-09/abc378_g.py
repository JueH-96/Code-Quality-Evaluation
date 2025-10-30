def solve():
    import sys
    input = sys.stdin.read
    from sys import stdin, stdout
    import math
    
    # Read input
    A, B, M = map(int, input().split())
    
    # Calculate AB
    AB = A * B
    
    # Factorial and modular inverse factorial precomputation
    factorial = [1] * (AB)
    inv_factorial = [1] * (AB)
    
    # Compute all factorials % M
    for i in range(2, AB):
        factorial[i] = factorial[i-1] * i % M
    
    # Compute all inverse factorials % M using Fermat's little theorem
    inv_factorial[AB-1] = pow(factorial[AB-1], M-2, M)
    for i in range(AB-2, 0, -1):
        inv_factorial[i] = inv_factorial[i+1] * (i+1) % M
    
    # Function to compute nCr % M
    def nCr(n, r):
        if r > n or r < 0:
            return 0
        return factorial[n] * inv_factorial[r] % M * inv_factorial[n-r] % M
    
    # Dynamic programming table
    # dp[x][y] will be the number of ways to arrange x+y elements such that:
    # - Longest increasing subsequence has length x
    # - Longest decreasing subsequence has length y
    dp = [[0] * (B+1) for _ in range(A+1)]
    dp[1][1] = 1
    
    # Fill the dp table
    for n in range(2, AB):
        new_dp = [[0] * (B+1) for _ in range(A+1)]
        for x in range(1, min(A, n)+1):
            for y in range(1, min(B, n)+1):
                if dp[x][y] == 0:
                    continue
                # Case 1: New element extends an increasing sequence
                new_dp[x][y+1] = (new_dp[x][y+1] + dp[x][y] * y) % M
                # Case 2: New element extends a decreasing sequence
                new_dp[x+1][y] = (new_dp[x+1][y] + dp[x][y] * x) % M
                # Case 3: New element is a new peak (increases both sequences)
                new_dp[x][y] = (new_dp[x][y] + dp[x][y] * (n - x - y + 1)) % M
        dp = new_dp
    
    # The result is the number of valid permutations of length AB-1
    result = dp[A][B]
    
    # Output the result
    stdout.write(str(result) + '
')