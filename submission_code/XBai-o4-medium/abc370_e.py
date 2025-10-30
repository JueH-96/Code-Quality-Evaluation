import sys
from collections import defaultdict

MOD = 998244353

def main():
    N, K = map(int, sys.stdin.readline().split())
    A = list(map(int, sys.stdin.readline().split()))
    prefix_sum = 0
    prefix_map = defaultdict(int)
    prefix_map[prefix_sum] = 1
    sum_dp = 1
    current_dp = 0
    for a in A:
        prefix_sum += a
        target = prefix_sum - K
        bad = prefix_map.get(target, 0)
        current_dp = (sum_dp - bad) % MOD
        sum_dp = (sum_dp + current_dp) % MOD
        prefix_map[prefix_sum] = (prefix_map.get(prefix_sum, 0) + current_dp) % MOD
    print(current_dp % MOD)

if __name__ == "__main__":
    main()