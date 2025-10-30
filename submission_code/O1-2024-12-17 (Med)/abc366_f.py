def main():
    import sys
    from functools import cmp_to_key
    
    input_data = sys.stdin.read().strip().split()
    N, K = map(int, input_data[:2])
    AB = list(zip(map(int, input_data[2::2]), map(int, input_data[3::2])))
    
    # Custom comparator for deciding which function should go "outside"
    # We say function x < y (x should be placed before y) if
    #   B_y * (A_x - 1) > B_x * (A_y - 1).
    # This ensures that when picked in ascending index order, x is outside y.
    def compare(x, y):
        A1, B1 = x
        A2, B2 = y
        left = B2 * (A1 - 1)
        right = B1 * (A2 - 1)
        # If left > right, x should come before y ("less" in Python's sorting sense)
        if left > right:
            return -1
        elif left < right:
            return 1
        else:
            return 0
    
    # Sort all functions in the optimal "outside to inside" order
    AB.sort(key=cmp_to_key(compare))
    
    # We'll do a 1D DP over k = 0..K while scanning the sorted list.
    # dp[k] = [alpha, beta, alpha_plus_beta] for picking k functions so far.
    # alpha, beta define the linear function c(x) = alpha * x + beta after composition,
    # and alpha_plus_beta = c(1) = alpha + beta.
    #
    # Initialize:
    dp = [[-1, -1, -1] for _ in range(K+1)]
    # If we pick 0 functions, that means alpha=1, beta=0 => final value alpha+beta=1
    dp[0] = [1, 0, 1]
    
    for (A, B) in AB:
        # Traverse k from K down to 1 so as not to overwrite states prematurely
        for k in range(K, 0, -1):
            if dp[k-1][2] >= 0:  # There's a valid way to have k-1 functions
                old_alpha, old_beta, _ = dp[k-1]
                pick_alpha = old_alpha * A
                pick_beta = old_alpha * B + old_beta
                pick_val = pick_alpha + pick_beta
                if pick_val > dp[k][2]:
                    dp[k] = [pick_alpha, pick_beta, pick_val]
    
    # The answer is dp[K][2], i.e. alpha + beta for the best choice of K functions
    print(dp[K][2])