import collections

def solve():
    N, W = map(int, input().split())
    items = []
    for _ in range(N):
        items.append(list(map(int, input().split())))

    # dp[j] will store the maximum total happiness for a total weight of j.
    # Initialize with -infinity, except dp[0] which is 0 (0 weight, 0 happiness).
    dp = [-float('inf')] * (W + 1)
    dp[0] = 0

    for w_i, v_i in items:
        # Create a new DP array for this item type, initialized with previous DP values.
        # This ensures that when we update dp_new[j], we are only combining with states
        # from *before* considering the current item type, which is necessary for the
        # multiple choice knapsack variant.
        dp_new = list(dp)

        # K_overall_max is the absolute maximum count of items of type `i` we might consider.
        # It's bounded by v_i (for non-negative happiness) and W // w_i (for weight capacity).
        K_overall_max = min(v_i, W // w_i)

        # Iterate over possible remainders when dividing by w_i
        for r in range(w_i):
            # Deque for monotonic queue optimization.
            # It will store quotients `q_prime` such that `score(q_prime)` is in decreasing order.
            # score(q_prime) = dp[q_prime * w_i + r] - q_prime * v_i + q_prime^2
            dq = collections.deque()

            # Iterate over quotients `q` for the current remainder `r`.
            # `j = q * w_i + r` is the current total weight being considered.
            for q in range((W - r) // w_i + 1):
                j = q * w_i + r 

                # 1. Remove elements from the front of the deque that are too "old".
                # An element `dq[0]` (representing `q_prev`) is too old if choosing
                # `k_chosen = q - dq[0]` items of type `i` would exceed `K_overall_max`.
                # This means `q - dq[0] > K_overall_max`, or `dq[0] < q - K_overall_max`.
                while dq and dq[0] < q - K_overall_max:
                    dq.popleft()

                # 2. Add the current `q` to the deque, maintaining the monotonic property of `score`.
                # Only consider `dp` states that are reachable (not -inf).
                # `dp[j]` refers to the value from the *previous* iteration (before current item type `i`).
                if dp[j] != -float('inf'):
                    current_score = dp[j] - q * v_i + q * q
                    # Remove elements from the back of the deque whose scores are no better than `current_score`.
                    while dq and (dp[dq[-1] * w_i + r] - dq[-1] * v_i + dq[-1] * dq[-1]) <= current_score:
                        dq.pop()
                    dq.append(q)
                
                # 3. Update `dp_new[j]` using the best candidate from the deque.
                if dq:
                    best_q_prev = dq[0] # This `q_prev` gives the maximum `score`.
                    
                    # Calculate `k_chosen`, the number of items of type `i` chosen.
                    k_chosen = q - best_q_prev
                    
                    # Calculate happiness contributed by choosing `k_chosen` items of type `i`.
                    happiness_from_k = k_chosen * v_i - k_chosen * k_chosen
                    
                    # If the base state `dp[best_q_prev * w_i + r]` was reachable, update `dp_new[j]`.
                    if dp[best_q_prev * w_i + r] != -float('inf'):
                        dp_new[j] = max(dp_new[j], dp[best_q_prev * w_i + r] + happiness_from_k)
        
        # After processing all remainders for the current item type,
        # update `dp` to `dp_new` for the next iteration.
        dp = dp_new

    # The maximum happiness is the largest value in the `dp` array.
    # If all values are still -inf, it means no positive happiness can be achieved.
    # In that case, the answer is 0 (by choosing no items).
    ans = max(dp)
    print(ans if ans != -float('inf') else 0)

solve()