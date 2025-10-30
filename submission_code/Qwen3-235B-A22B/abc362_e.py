import sys
from collections import defaultdict

MOD = 998244353

def main():
    input = sys.stdin.read().split()
    n = int(input[0])
    A = list(map(int, input[1:n+1]))
    
    if n == 0:
        print()
        return
    
    # Initialize dp: dp[k][i] is a defaultdict(int) for each k >=2
    max_k = n
    dp = [ [defaultdict(int) for _ in range(n)] for _ in range(n + 2) ]  # dp[0], dp[1] unused or ignored

    # Fill dp[2]
    for i in range(n):
        for j in range(i):
            d = A[i] - A[j]
            dp[2][i][d] = (dp[2][i][d] + 1) % MOD
    
    # Fill dp for k >=3 up to n
    for k in range(3, n + 1):
        for i in range(n):
            for j in range(i):
                d = A[i] - A[j]
                prev = dp[k-1][j][d]
                if prev:
                    dp[k][i][d] = (dp[k][i][d] + prev) % MOD
    
    # Prepare the answer
    ans = []
    # k=1 case
    ans.append(n % MOD)
    
    # k from 2 to n
    for k in range(2, n + 1):
        total = 0
        for i in range(n):
            current_sum = sum(dp[k][i].values())
            total = (total + current_sum) % MOD
        ans.append(total)
    
    print(' '.join(map(str, ans)))

if __name__ == "__main__":
    main()