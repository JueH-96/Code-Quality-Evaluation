class Solution:
    def findKthSmallest(self, coins: List[int], k: int) -> int:
        import math
        from functools import lru_cache

        # We want the k-th smallest number in the union of all multiples of each coin.
        # We cannot combine different coins together; each amount is purely a multiple of one coin.
        #
        # In other words, our set is:
        #   { c*m | c in coins, m >= 1 }
        # We seek the k-th smallest distinct value in that infinite union.
        #
        # Observing that k can be as large as 2e9, generating multiples directly would be infeasible.
        # Instead, we can use a binary search on the amount x and ask:
        #   "How many distinct multiples are <= x across all coins?"
        # We denote this as f(x).
        #
        # If f(mid) >= k, we can move the high bound down. Otherwise, we move the low bound up.
        # Finally, when low meets high, that is the k-th smallest.
        #
        # The challenge is to compute f(x) efficiently without double-counting numbers that are multiples
        # of more than one coin. We can use the inclusion-exclusion principle:
        #
        #   f(x) = sum_{S ⊆ coins, S nonempty} ((-1)^(|S|+1) * floor(x / lcm(S))),
        #
        #   where lcm(S) is the least common multiple of the coins in subset S, and |S| is its size.
        #
        # Because coins.length <= 15, the number of non-empty subsets is at most 2^15 - 1 = 32767,
        # which is large but we can handle it with some care (e.g. precomputing lcms).
        #
        # We use:
        #   - min_coin = min(coins)
        #   - the search range for x is [1..min_coin*k], because the k-th smallest cannot exceed k multiples of the smallest coin.

        coins = list(coins)
        n = len(coins)
        min_coin = min(coins)

        # Precompute all subsets' LCM using DFS (or bitmask), skipping those where LCM exceeds min_coin*k
        # because we won't need them in our queries (they won't contribute to counts <= x in our search range).

        limit = min_coin * k  # maximum we might need to test

        # We'll store tuples of (lcm_value, sign) where sign = +1 if subset size is odd, -1 if even.
        lcms = []

        # Recursive function to build subsets
        def build_subsets(idx, current_lcm, size):
            if idx == n:
                if size > 0:
                    # Subset is not empty; sign depends on the subset's size
                    sign = 1 if (size % 2) == 1 else -1
                    lcms.append((current_lcm, sign))
                return
            # Option 1: skip this coin
            build_subsets(idx + 1, current_lcm, size)
            # Option 2: include this coin
            if current_lcm == 0:
                new_lcm = coins[idx]
            else:
                # lcm(a,b) = (a*b // gcd(a,b)), watch for overflow in Python (though Python can handle big int)
                gcd_val = math.gcd(current_lcm, coins[idx])
                new_lcm = (current_lcm // gcd_val) * coins[idx]
            if new_lcm <= limit:  # only recurse further if it doesn't exceed our maximum needed
                build_subsets(idx + 1, new_lcm, size + 1)

        build_subsets(0, 0, 0)

        # Function f(x) to count how many distinct multiples are <= x
        def count_distinct_multiples_up_to(x):
            total = 0
            for val, sgn in lcms:
                if val <= x:
                    total += sgn * (x // val)
            return total

        # Binary search for the smallest x such that f(x) >= k
        low, high = 1, limit
        while low < high:
            mid = (low + high) // 2
            if count_distinct_multiples_up_to(mid) >= k:
                high = mid
            else:
                low = mid + 1

        return low