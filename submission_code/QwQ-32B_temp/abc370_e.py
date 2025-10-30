import sys
from collections import defaultdict

def main():
    mod = 998244353
    N, K = map(int, sys.stdin.readline().split())
    A = list(map(int, sys.stdin.readline().split()))
    
    current_S = 0
    sum_counts = defaultdict(int)
    sum_counts[0] = 1  # DP[0] is 1
    total_sum = 1 % mod
    ans = 1  # DP[0]
    
    for i in range(1, N+1):
        a = A[i-1]
        current_S += a
        target = current_S - K
        subtract = sum_counts.get(target, 0)
        dp_i = (total_sum - subtract) % mod
        # Update sum_counts for current_S
        sum_counts[current_S] = (sum_counts.get(current_S, 0) + dp_i) % mod
        # Update total_sum
        total_sum = (total_sum + dp_i) % mod
        ans = dp_i
    
    print(ans % mod)

if __name__ == "__main__":
    main()