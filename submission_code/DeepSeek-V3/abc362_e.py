# YOUR CODE HERE
import sys

MOD = 998244353

def main():
    N, *rest = map(int, sys.stdin.read().split())
    A = rest[:N]
    
    # Initialize dp[k][d][last] where k is the length of the subsequence, d is the common difference, and last is the last element
    # We will use a dictionary to represent the dp table for each k
    # Initialize dp[1][d][A[i]] = 1 for each i
    dp = [dict() for _ in range(N+1)]
    for i in range(N):
        if 1 not in dp:
            dp[1] = dict()
        if A[i] not in dp[1]:
            dp[1][A[i]] = 0
        dp[1][A[i]] += 1
    
    for k in range(2, N+1):
        dp[k] = dict()
        for prev_d in dp[k-1]:
            for last in dp[k-1][prev_d]:
                for i in range(N):
                    if A[i] > last:
                        d = A[i] - last
                        if d == prev_d:
                            if d not in dp[k]:
                                dp[k][d] = dict()
                            if A[i] not in dp[k][d]:
                                dp[k][d][A[i]] = 0
                            dp[k][d][A[i]] += dp[k-1][prev_d][last]
                            dp[k][d][A[i]] %= MOD
    
    # Now, for each k, sum all the counts in dp[k]
    result = []
    for k in range(1, N+1):
        total = 0
        if k in dp:
            for d in dp[k]:
                for last in dp[k][d]:
                    total += dp[k][d][last]
                    total %= MOD
        result.append(total % MOD)
    
    print(' '.join(map(str, result)))

if __name__ == "__main__":
    main()