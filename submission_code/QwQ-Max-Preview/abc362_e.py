import sys
from collections import defaultdict

MOD = 998244353

def main():
    n = int(sys.stdin.readline())
    A = list(map(int, sys.stdin.readline().split()))
    
    if n == 0:
        return
    
    # dp[i] is a dictionary mapping difference d to a dictionary mapping length k to count
    dp = [defaultdict(lambda: defaultdict(int)) for _ in range(n)]
    
    for i in range(n):
        for j in range(i):
            d = A[i] - A[j]
            # Add the pair (j, i) as a new subsequence of length 2
            dp[i][d][2] += 1
            # Check if there are existing subsequences ending at j with the same difference
            if d in dp[j]:
                for k in dp[j][d]:
                    new_k = k + 1
                    dp[i][d][new_k] = (dp[i][d][new_k] + dp[j][d][k]) % MOD
    
    # Prepare the answer array
    ans = [0] * (n + 1)  # ans[0] unused
    ans[1] = n  # all single elements are valid
    
    for k in range(2, n + 1):
        total = 0
        for i in range(n):
            for d in dp[i]:
                if k in dp[i][d]:
                    total = (total + dp[i][d][k]) % MOD
        ans[k] = total
    
    # Output the results from ans[1] to ans[n]
    print(' '.join(map(str, ans[1:n+1])))

if __name__ == '__main__':
    main()