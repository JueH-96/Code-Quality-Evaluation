import sys
import bisect

def solve():
    N, M, P = map(int, sys.stdin.readline().split())
    A_prices = list(map(int, sys.stdin.readline().split()))
    B_prices = list(map(int, sys.stdin.readline().split()))

    B_prices.sort()

    # prefix_sum_B[k] will store sum of the first k elements of sorted B_prices
    # i.e., B_prices[0] + ... + B_prices[k-1]
    prefix_sum_B = [0] * (M + 1)
    for i in range(M):
        prefix_sum_B[i+1] = prefix_sum_B[i] + B_prices[i]

    total_overall_price = 0

    for a_val in A_prices:
        # For a fixed main dish price a_val:
        # A set meal (a_val, b_val) costs a_val + b_val if a_val + b_val < P,
        # otherwise it costs P.
        # This is equivalent to:
        # - If b_val < P - a_val, cost is a_val + b_val.
        # - If b_val >= P - a_val, cost is P.
        
        target_b_threshold = P - a_val
        
        # k_count is the number of side dishes b_val such that b_val < target_b_threshold.
        # These are B_prices[0]...B_prices[k_count-1].
        k_count = bisect.bisect_left(B_prices, target_b_threshold)
        
        # Sum of prices for meals with b_val < target_b_threshold:
        # Sum (a_val + B_prices[j] for j from 0 to k_count-1)
        # = k_count * a_val + (B_prices[0] + ... + B_prices[k_count-1])
        # = k_count * a_val + prefix_sum_B[k_count]
        sum_for_smaller_b = k_count * a_val + prefix_sum_B[k_count]
        
        # Sum of prices for meals with b_val >= target_b_threshold:
        # There are (M - k_count) such side dishes. Each meal costs P.
        # Sum = (M - k_count) * P
        sum_for_larger_b = (M - k_count) * P
        
        total_overall_price += sum_for_smaller_b
        total_overall_price += sum_for_larger_b
        
    sys.stdout.write(str(total_overall_price) + "
")

if __name__ == '__main__':
    solve()