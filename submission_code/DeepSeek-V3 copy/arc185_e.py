import sys
import math

MOD = 998244353

def main():
    input = sys.stdin.read
    data = input().split()
    N = int(data[0])
    A = list(map(int, data[1:N+1]))
    
    # Initialize dp and sum_dp
    dp = {}  # dp[g] will store the number of subsequences ending with g
    sum_dp = {}  # sum_dp[g] will store the sum of scores for subsequences ending with g
    total = 0  # total sum of scores for all subsequences
    
    for i in range(N):
        a = A[i]
        new_dp = {}
        new_sum_dp = {}
        
        # Initialize with the current element as a new subsequence
        new_dp[a] = 1
        new_sum_dp[a] = 0
        
        # Iterate over all existing subsequences
        for g in dp:
            cnt = dp[g]
            s = sum_dp[g]
            new_g = math.gcd(g, a)
            # Update new_dp and new_sum_dp
            if new_g in new_dp:
                new_dp[new_g] += cnt
                new_sum_dp[new_g] += s + cnt * math.gcd(g, a)
            else:
                new_dp[new_g] = cnt
                new_sum_dp[new_g] = s + cnt * math.gcd(g, a)
            # Ensure modulo operation
            new_dp[new_g] %= MOD
            new_sum_dp[new_g] %= MOD
        
        # Merge new_dp and new_sum_dp into dp and sum_dp
        for g in new_dp:
            if g in dp:
                dp[g] += new_dp[g]
                sum_dp[g] += new_sum_dp[g]
            else:
                dp[g] = new_dp[g]
                sum_dp[g] = new_sum_dp[g]
            dp[g] %= MOD
            sum_dp[g] %= MOD
        
        # Calculate the total sum for the current m
        current_total = 0
        for g in sum_dp:
            current_total += sum_dp[g]
            current_total %= MOD
        print(current_total)

if __name__ == "__main__":
    main()