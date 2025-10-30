class Solution:
    MOD = 10**9 + 7
    max_fact = 10**5
    fact = None
    inv_fact = None

    def __init__(self):
        if Solution.fact is None or Solution.inv_fact is None:
            # Precompute factorials and inverse factorials up to max_fact
            Solution.fact = [1] * (Solution.max_fact + 1)
            for i in range(1, Solution.max_fact + 1):
                Solution.fact[i] = Solution.fact[i-1] * i % Solution.MOD
            # Compute inverse factorials
            Solution.inv_fact = [1] * (Solution.max_fact + 1)
            Solution.inv_fact[Solution.max_fact] = pow(Solution.fact[Solution.max_fact], Solution.MOD - 2, Solution.MOD)
            for i in range(Solution.max_fact - 1, -1, -1):
                Solution.inv_fact[i] = Solution.inv_fact[i + 1] * (i + 1) % Solution.MOD

    def numberOfSequence(self, n: int, sick: List[int]) -> int:
        MOD = self.MOD
        gaps = []
        prev = -1

        # Process left and middle gaps
        for s in sick:
            if prev == -1:
                left_length = s - 0
                if left_length > 0:
                    gaps.append((left_length, 'left'))
            else:
                mid_length = s - prev - 1
                if mid_length > 0:
                    gaps.append((mid_length, 'middle'))
            prev = s

        # Process right gap
        if sick:
            last = sick[-1]
            right_length = (n - 1) - last
            if right_length > 0:
                gaps.append((right_length, 'right'))

        product_ways = 1
        sum_len = 0
        inv_fact_product = 1

        for (m, gap_type) in gaps:
            if gap_type == 'middle':
                pow_2 = pow(2, m - 1, MOD)
                product_ways = product_ways * pow_2 % MOD
            sum_len += m
            inv_fact_product = inv_fact_product * self.inv_fact[m] % MOD

        # Calculate multinomial coefficient
        if sum_len == 0:
            return product_ways
        multinom = self.fact[sum_len] * inv_fact_product % MOD
        total = product_ways * multinom % MOD
        return total