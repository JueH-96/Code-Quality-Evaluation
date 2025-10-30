def main():
    import sys
    input_data = sys.stdin.read().strip().split()
    N, W = map(int, input_data[:2])
    items = list(zip(
        map(int, input_data[2::2]),
        map(int, input_data[3::2])
    ))

    # dp[x] will hold the maximum "happiness" achievable with total weight exactly x
    dp = [0] * (W + 1)

    for w_i, v_i in items:
        # We'll create new_dp from dp, representing states after considering this item type
        new_dp = dp[:]  # make a copy

        # The maximum count k we might try for this item (beyond that, happiness won't improve)
        # 1) We can't exceed W // w_i by capacity.
        # 2) We won't exceed v_i, because at k = v_i+1, happiness would become negative.
        max_k = min(W // w_i, v_i)

        # For each current weight x, try adding k items of this type
        for x in range(W + 1):
            base_val = dp[x]  # best we had with total weight x before using this type
            # Now try 1..max_k items so long as we stay within the capacity
            # We'll accumulate the "weight cost" and "happiness" incrementally
            weight_sum = 0
            happiness_sum = 0
            for k in range(1, max_k + 1):
                weight_sum += w_i
                if x + weight_sum > W:
                    break
                # happiness for picking k items of this type is k*v_i - k^2,
                # but accumulate incrementally to avoid repeated multiplication
                #    k*v_i - k^2 = (k-1)*v_i - (k-1)^2 + [v_i - (2*(k-1)+1)]
                # We'll just do it directly for clarity:
                # happiness_sum = k*v_i - k^2
                # But to do it incrementally:
                #   next increment = v_i - (2*(k-1)+1)
                # For simplicity, compute directly each time:
                happiness_sum = k*v_i - k*k
                candidate_val = base_val + happiness_sum
                if candidate_val > new_dp[x + weight_sum]:
                    new_dp[x + weight_sum] = candidate_val

        dp = new_dp

    print(max(dp))

# Call main() to execute
main()