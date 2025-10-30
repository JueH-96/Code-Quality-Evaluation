class Solution:
    MOD = 10**9 + 7
    max_n = 10**5
    fact = None
    inv_fact = None

    def __init__(self):
        if Solution.fact is None:
            self.precompute()

    def precompute(self):
        MOD = self.MOD
        max_n = self.max_n
        fact = [1] * (max_n + 1)
        for i in range(1, max_n + 1):
            fact[i] = fact[i-1] * i % MOD
        inv_fact = [1] * (max_n + 1)
        inv_fact[max_n] = pow(fact[max_n], MOD-2, MOD)
        for i in range(max_n - 1, -1, -1):
            inv_fact[i] = inv_fact[i+1] * (i+1) % MOD
        Solution.fact = fact
        Solution.inv_fact = inv_fact

    def numberOfSequence(self, n: int, sick: List[int]) -> int:
        MOD = self.MOD
        self.precompute()  # Ensure fact and inv_fact are precomputed

        intervals = []
        prev = -1
        for s in sick:
            start = prev + 1
            end = s - 1
            if start <= end:
                interval_type = 'left' if prev == -1 else 'middle'
                intervals.append((start, end, interval_type))
            prev = s

        # Add the rightmost interval
        start = prev + 1
        end = n - 1
        if start <= end:
            intervals.append((start, end, 'right'))

        product_ways = 1
        product_denominator = 1
        m = 0

        for (start, end, interval_type) in intervals:
            k = end - start + 1
            if interval_type == 'middle':
                ways = pow(2, k-1, MOD)
            else:
                ways = 1
            product_ways = (product_ways * ways) % MOD
            product_denominator = (product_denominator * Solution.inv_fact[k]) % MOD
            m += k

        if m == 0:
            return 1  # All children were already sick

        multinomial = (Solution.fact[m] * product_denominator) % MOD
        total = (product_ways * multinomial) % MOD
        return total