class Solution:
    def findKthSmallest(self, coins: List[int], k: int) -> int:
        import math
        coins.sort()

        # We will use binary search on the answer range.
        # Let low = 1, high = min(coins) * k (worst case: the smallest coin used k times).
        # Then we define a function count_up_to(x) that returns how many distinct multiples
        # are <= x when considering all coins' multiples (using inclusion-exclusion).

        def gcd(a, b):
            # math.gcd could be used directly, but defining here for clarity
            while b:
                a, b = b, a % b
            return a

        def lcm(a, b):
            return a // gcd(a, b) * b

        def count_up_to(x):
            # Returns how many distinct multiples are <= x by inclusion-exclusion.
            # We do a DFS over subsets of coins. For each subset, we add or subtract
            # floor(x / LCM(subset)) appropriately based on the subset size (odd/even length).
            n = len(coins)
            total = 0

            def dfs(start_idx, current_lcm, sign):
                nonlocal total
                for i in range(start_idx, n):
                    new_lcm = lcm(current_lcm, coins[i])
                    if new_lcm > x:
                        # If LCM already exceeds x, further picking larger coins won't help.
                        break
                    total += sign * (x // new_lcm)
                    # flip the sign for the next level (inclusion-exclusion)
                    dfs(i + 1, new_lcm, -sign)

            # Start DFS with an empty subset (lcm=1) and sign=1
            dfs(0, 1, 1)
            return total

        low, high = 1, coins[0] * k

        while low < high:
            mid = (low + high) // 2
            if count_up_to(mid) >= k:
                high = mid
            else:
                low = mid + 1

        return low