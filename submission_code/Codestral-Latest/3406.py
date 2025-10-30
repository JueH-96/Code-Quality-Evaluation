class Solution:
    def numberOfStableArrays(self, zero: int, one: int, limit: int) -> int:
        MOD = 10**9 + 7

        # Helper function to calculate binomial coefficient
        def binomial_coefficient(n, k):
            if k > n:
                return 0
            if k == 0 or k == n:
                return 1
            k = min(k, n - k)
            c = 1
            for i in range(k):
                c = c * (n - i) // (i + 1)
            return c

        # Total number of ways to arrange zero 0s and one 1
        total_ways = binomial_coefficient(zero + one, one)

        # Function to check if a given arrangement is stable
        def is_stable(arrangement):
            n = len(arrangement)
            for i in range(n):
                for j in range(i + limit + 1, n + 1):
                    subarray = arrangement[i:j]
                    if all(x == 0 for x in subarray) or all(x == 1 for x in subarray):
                        return False
            return True

        # Generate all possible arrangements and count the stable ones
        stable_count = 0
        from itertools import combinations
        for combo in combinations(range(zero + one), one):
            arrangement = [0] * (zero + one)
            for idx in combo:
                arrangement[idx] = 1
            if is_stable(arrangement):
                stable_count += 1

        return stable_count % MOD