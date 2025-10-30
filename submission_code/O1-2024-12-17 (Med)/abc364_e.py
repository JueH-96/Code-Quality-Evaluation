def main():
    import sys
    input_data = sys.stdin.read().strip().split()
    N, X, Y = map(int, input_data[:3])
    dishes = list(zip(
        map(int, input_data[3::2]),
        map(int, input_data[4::2])
    ))

    # ----------------------------------------------------------------
    # Explanation of the approach:
    #
    # Let us interpret the eating rule carefully:
    #  - The dishes can be arranged in any order we like.
    #  - Snuke starts from the first dish in that order and goes one by one.
    #  - At the moment Snuke is about to start dish i (1-based), we check the
    #    total sweetness and saltiness so far (i - 1 dishes eaten). If that
    #    partial sum is at most (X, Y), then Snuke does eat dish i.
    #  - After actually eating dish i, if the new sum exceeds X or Y, Snuke
    #    stops and does not proceed to dish i+1. However, dish i is still
    #    counted as eaten.
    #
    # In simpler terms:
    #  - You get to place dishes in a sequence.
    #  - For i = 1..k, you require that the sum of the first i-1 dishes is <= (X,Y),
    #    so that you are allowed to eat dish i.
    #  - The sum after eating dish i can exceed (X,Y), in which case you stop,
    #    but you still count dish i.
    #
    # Therefore, to maximize the count of dishes eaten, we want to find a sequence
    # (d1, d2, ..., dK) such that for every i in 1..K, the sum of A(d1)+...+A(d(i-1)) <= X
    # and the sum of B(d1)+...+B(d(i-1)) <= Y.  (Where the sum of the first 0 dishes is 0.)
    # The partial sum after eating the K-th dish may or may not exceed (X,Y).
    #
    # Another way: the only real restriction is that all prefixes (up to K-1 dishes)
    # stay within (X,Y).  Then dish K can be large; we will still get to eat it.
    #
    # We will solve it by a dynamic-programming–with–"dominance"-compression approach:
    #
    # Let dp be a dictionary mapping (sweet_sum, salt_sum) -> the maximum number
    # of dishes that can have *that* exact partial sum in some chosen subset/ordering
    # *without ever having exceeded the limit in an earlier prefix*.  In other words,
    # these states represent valid partial sums so far (which are ≤(X,Y)) in some order.
    #
    # We also keep a separate variable dp_ex which tracks the maximum number of dishes
    # if we "exceed" right after adding some dish.  Because once we exceed,
    # we cannot add any more dishes, but we do count the dish that caused the exceed.
    #
    # Algorithm sketch:
    #  1) Initialize dp = {(0,0): 0}, meaning with sum=(0,0) we have eaten 0 dishes.
    #  2) For each dish i in [1..N] (in any fixed order, e.g. input order):
    #        - We create a copy newdp = dp (so we also have the option "skip dish i"),
    #        - For each (a,b)->count in dp:
    #             a' = a + A_i
    #             b' = b + B_i
    #             if a' <= X and b' <= Y:
    #                 # We remain within the limit, so partial sum is still valid
    #                 newdp[a', b'] = max(newdp.get((a', b'), 0), count+1)
    #             else:
    #                 # We exceed the limit upon eating this dish,
    #                 # so we stop there, but we do count this dish
    #                 dp_ex = max(dp_ex, count+1)
    #        - Then dp = a "minimized" version of newdp where we remove dominated states.
    #
    # The "minimizing" or "dominance" step means:
    #  - If we have (a1, b1) with count1, and (a2, b2) with count2,
    #    and a2 <= a1, b2 <= b1, count2 >= count1, then (a1,b1) is "dominated"
    #    and we can discard it from dp.  Because that means for partial sums
    #    up to dishes, (a2,b2) is strictly as good or better than (a1,b1),
    #    so (a1,b1) will never help us form a better arrangement.
    #
    #  - Keeping the dp dictionary from exploding is crucial to make this feasible.
    #    In worst cases it could still be large, but in practice (and for test data)
    #    this approach tends to remain within a reasonable size.
    #
    # After processing all dishes, the maximum number we can get without ever exceeding
    # is max(dp.values()); but we also might "exceed" after the last dish added,
    # which is tracked by dp_ex.  So the final answer is max( max(dp.values()), dp_ex ).
    #
    # That solves the problem.  We just have to implement it carefully.
    # ----------------------------------------------------------------

    from collections import defaultdict

    # A function to perform the "dominance minimization" step on dp.
    # dp is a dict: (a, b) -> best_count
    # We'll discard any (a1, b1) that is dominated by some (a2, b2) with
    # a2 <= a1, b2 <= b1, and dp[a2,b2] >= dp[a1,b1].
    #
    # Implementation strategy:
    #  1. Extract items: (a, b, c) where c = dp[a,b].
    #  2. Sort them primarily by a ascending, then by b ascending, then by c descending.
    #  3. We'll do a "sweep" and keep track of the best c encountered so far in
    #     a monotonic way in the dimension b.  But we must be careful to do a
    #     2D version.  A standard approach is:
    #       - Sort by a ascending, then by b ascending, c descending
    #       - Traverse from smallest (a,b) to largest
    #       - We maintain a structure for the best c we've seen so far for
    #         smaller or equal b.  If the current (a,b,c) has c <= that best c,
    #         it is dominated.
    #
    #  Because we are sorting in ascending b as well, once we move forward in
    #  that sorted list, we never come back to a smaller b.  We just track
    #  the maximum c so far.  If the current c is <= that maximum c, we are dominated.
    #  If c is bigger, then we keep it and update the max c.  But that takes care
    #  of exactly the situation a2 <= a1 and b2 <= b1.  (Because we are going in
    #  ascending order of a and b, any previously visited item has a2 <= a1, b2 <= b1.)
    #
    #  Let's implement that carefully.
    #
    def remove_dominated(dp_dict):
        # Turn dict into a list of (a, b, c)
        items = [(xy[0], xy[1], dp_dict[xy]) for xy in dp_dict]
        # Sort: a ascending, b ascending, c descending
        items.sort(key=lambda x: (x[0], x[1], -x[2]))

        res = {}
        max_c_so_far = -1
        # We'll sweep left-to-right in this sorted order.
        # Because (a,b) is non-decreasing in this order,
        # any future item has a >= current.a, b >= current.b.
        # So if we already have a previous item with c' >= c,
        # then the current item is dominated.
        #
        # Implementation detail: if (a,b) are the same for multiple items,
        # we only keep the largest c among them anyway in the dictionary. So
        # we won't store multiple.  But let's just handle it robustly.
        for a, b, c in items:
            if c > max_c_so_far:
                # keep this one
                res[(a,b)] = c
                max_c_so_far = c
                # Because b is non-decreasing, once we have c = something,
                # any next item with the same or bigger (a,b) but c <= that
                # is dominated.
        return res

    dp = {(0,0): 0}  # partial_sum -> best count
    dp_ex = 0        # best count if we exceed immediately after adding a dish

    for (sweet, salt) in dishes:
        newdp = dict(dp)  # start by copying old states (the "skip dish" option)

        # Try to take the current dish from each feasible state
        for (a, b), count in dp.items():
            # we "eat" this dish from partial sum (a,b)
            c_new = count + 1
            a_new = a + sweet
            b_new = b + salt
            # If that surpasses X or Y, we record that we have just exceeded
            # and can't eat more, but we do get this dish.
            if a_new > X or b_new > Y:
                if c_new > dp_ex:
                    dp_ex = c_new
            else:
                # still within limit, update newdp
                old_val = newdp.get((a_new, b_new), 0)
                if c_new > old_val:
                    newdp[(a_new, b_new)] = c_new

        # Now remove dominated states in newdp
        dp = remove_dominated(newdp)

    # The best result if we never exceed is max(dp.values())
    best_in_dp = max(dp.values()) if dp else 0

    # The final answer is the maximum of:
    #  -- eating only dishes that never made partial sums exceed
    #  -- or exceeding after the last dish that caused the exceed
    ans = max(best_in_dp, dp_ex)
    print(ans)

# Don't forget to call main()!
if __name__ == "__main__":
    main()