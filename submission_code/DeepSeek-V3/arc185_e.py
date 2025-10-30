import sys
import math

MOD = 998244353

def main():
    input = sys.stdin.read
    data = input().split()
    N = int(data[0])
    A = list(map(int, data[1:N+1]))
    
    # Initialize dp and sum_dp
    dp = [0] * (10**5 + 1)
    sum_dp = 0
    total = 0
    
    for i in range(N):
        a = A[i]
        new_dp = [0] * (10**5 + 1)
        # For each possible gcd value in dp
        for g in range(1, 10**5 + 1):
            if dp[g]:
                new_gcd = math.gcd(g, a)
                new_dp[new_gcd] = (new_dp[new_gcd] + dp[g]) % MOD
        # Add the new subsequence starting with a
        new_dp[a] = (new_dp[a] + 1) % MOD
        # Update sum_dp
        sum_dp = (sum_dp * 2) % MOD
        for g in range(1, 10**5 + 1):
            sum_dp = (sum_dp + dp[g] * g) % MOD
        # Update dp
        dp = new_dp
        # Update total
        total = (total * 2 + sum_dp) % MOD
        print(total)

if __name__ == "__main__":
    main()