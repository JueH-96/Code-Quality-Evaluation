import math

def main():
    import sys
    input = sys.stdin.read().split()
    n = int(input[0])
    p_list = list(map(int, input[1:n+1]))
    
    max_m = n
    INF = float('inf')
    dp = [-INF] * (max_m + 1)
    dp[0] = 0.0  # Base case: 0 elements selected (not allowed, but used for transitions)
    
    for p in p_list:
        temp_dp = dp.copy()
        for m in range(max_m + 1):
            if temp_dp[m] == -INF:
                continue
            new_m = m + 1
            if new_m > max_m:
                continue
            new_sum = temp_dp[m] * 0.9 + p
            if new_sum > dp[new_m]:
                dp[new_m] = new_sum
    
    max_r = -INF
    for m in range(1, max_m + 1):
        if dp[m] == -INF:
            continue
        denominator = 10.0 * (1.0 - (0.9 ** m))
        part1 = dp[m] / denominator
        penalty = 1200.0 / math.sqrt(m)
        current_r = part1 - penalty
        if current_r > max_r:
            max_r = current_r
    
    print("{0:.12f}".format(max_r))

if __name__ == '__main__':
    main()