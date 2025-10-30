# YOUR CODE HERE
import sys
input = sys.stdin.read

MOD = 998244353

def solve():
    data = input().split()
    N = int(data[0])
    A = list(map(int, data[1:N+1]))
    B = list(map(int, data[N+1:2*N+1]))
    
    # dp[i][j] will store the number of valid ways to fill the first i sequences
    # such that the first i elements of a are fixed and the first i elements of b are fixed
    dp = [[0] * (2 * N + 1) for _ in range(N + 1)]
    dp[0][0] = 1
    
    for i in range(1, N + 1):
        for j in range(2 * N + 1):
            if dp[i-1][j] == 0:
                continue
            
            for k in range(1, N + 1):
                if k in A[:i-1] or k in B[:i-1]:
                    continue
                
                if A[i-1] == j + 1:
                    if B[i-1] == -1:
                        for l in range(1, 2 * N + 1):
                            if l not in A[:i] and l not in B[:i-1]:
                                dp[i][l] = (dp[i][l] + dp[i-1][j]) % MOD
                    else:
                        if B[i-1] not in A[:i] and B[i-1] not in B[:i-1]:
                            dp[i][B[i-1]] = (dp[i][B[i-1]] + dp[i-1][j]) % MOD
                elif B[i-1] == j + 1:
                    if A[i-1] not in A[:i-1] and A[i-1] not in B[:i]:
                        dp[i][A[i-1]] = (dp[i][A[i-1]] + dp[i-1][j]) % MOD
    
    result = sum(dp[N]) % MOD
    print(result)

solve()