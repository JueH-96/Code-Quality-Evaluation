def solve():
    import sys
    input_data = sys.stdin.read().strip().split()
    N, X, Y = map(int, input_data[:3])
    dishes = []
    idx = 3
    for _ in range(N):
        A_i, B_i = map(int, input_data[idx:idx+2])
        idx += 2
        dishes.append((A_i, B_i))
    
    # We want to find the maximum number of dishes that can be eaten
    # if we arrange them optimally, subject to partial sums never exceeding X or Y.
    #
    # A known trick is to use a DP approach (similar to knapsack):
    #   - We do not actually need the exact order in which to place them.
    #   - We only need the best possible subset (and the possibility to order that subset)
    #     so that partial sums never exceed X or Y, which is equivalent (with careful ordering)
    #     to the total sweetness staying ≤ X and total saltiness staying ≤ Y at each step
    #     if it is possible at all.
    #
    # Concretely, we can use a DP that stores, for each number of dishes considered
    # (from the first i dishes) and each total sweetness s (0 ≤ s ≤ X),
    # the pair (max_count, min_salt) = 
    #   the maximum number of dishes used to achieve exactly sweetness s,
    #   and among those subsets that achieve that max_count, the minimal saltiness.
    #
    # After processing all dishes, we look over dp[N][s] for s in [0..X],
    # and check which of them has min_salt ≤ Y.  The best max_count among those is our answer.
    #
    # This DP works in O(N*X) time, which is feasible for N ≤ 80 and X ≤ 10000.
    #
    # Implementation details:
    #   dp[i][s] = (count_i, salt_i).
    #   We start with dp[0][0] = (0, 0) and dp[0][s>0] = (-∞, ∞).
    #   Transition when we consider the i-th dish (index i-1 in 0-based):
    #      not pick: dp[i][s] can come from dp[i-1][s].
    #      pick: if s-A_i >= 0, then we can do dp[i-1][s-A_i] -> dp[i][s]
    #        new_count = dp[i-1][s-A_i].count + 1
    #        new_salt  = dp[i-1][s-A_i].salt + B_i
    #      we compare new_count with current dp[i][s].count:
    #        if new_count is bigger, we update dp[i][s] = (new_count, new_salt)
    #        if new_count is equal, we pick the smaller salt.
    #   In the end, the answer is max{ dp[N][s].count | 0 ≤ s ≤ X and dp[N][s].salt ≤ Y }.

    # For convenience, let's just keep one DP array rolling over i (we only need i-1 to i).
    # dp[s] = (count, salt)
    # We'll update into a new array new_dp[s] for each dish.

    # Initialize DP
    # dp[s] = (count, salt). Use a list of tuples.
    dp = [(-1, float('inf'))] * (X + 1)
    dp[0] = (0, 0)  # 0 dishes used, 0 saltiness if sweetness=0

    for (A_i, B_i) in dishes:
        new_dp = dp[:]  # copy current dp
        for s in range(X + 1):
            count, salt = dp[s]
            if count < 0:
                continue  # impossible state
            # Not picking this dish: new_dp[s] can remain as is, no change needed

            # If we pick this dish:
            s2 = s + A_i
            if s2 <= X:
                new_count = count + 1
                new_salt = salt + B_i
                cur_count, cur_salt = new_dp[s2]
                if new_count > cur_count:
                    new_dp[s2] = (new_count, new_salt)
                elif new_count == cur_count and new_salt < cur_salt:
                    new_dp[s2] = (new_count, new_salt)
        dp = new_dp

    ans = 0
    for s in range(X + 1):
        count, salt = dp[s]
        if count >= 0 and salt <= Y:
            ans = max(ans, count)
    
    print(ans)

def main():
    solve()

if __name__ == "__main__":
    main()