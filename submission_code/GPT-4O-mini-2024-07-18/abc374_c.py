def min_max_lunch_break(N, K):
    total_people = sum(K)
    half_total = total_people // 2
    
    # We will use a dynamic programming approach to find the closest sum to half_total
    dp = [0] * (half_total + 1)
    
    for k in K:
        for j in range(half_total, k - 1, -1):
            dp[j] = max(dp[j], dp[j - k] + k)
    
    # The best we can do is dp[half_total], which is the closest sum to half_total
    group_a_sum = dp[half_total]
    group_b_sum = total_people - group_a_sum
    
    return max(group_a_sum, group_b_sum)

import sys
input = sys.stdin.read

def main():
    data = input().strip().split()
    N = int(data[0])
    K = list(map(int, data[1:N+1]))
    
    result = min_max_lunch_break(N, K)
    print(result)

if __name__ == "__main__":
    main()