class Solution:
    def findProductsOfElements(self, queries: List[List[int]]) -> List[int]:
        from itertools import chain
        
        def powerful_array(n):
            powers = []
            while n > 0:
                power = 1
                while power <= n:
                    power <<= 1
                power >>= 1
                powers.append(power)
                n -= power
            return powers
        
        # Generate big_nums for the first few integers
        big_nums = list(chain.from_iterable(powerful_array(i) for i in range(1, 100)))
        
        results = []
        for from_i, to_i, mod_i in queries:
            product = 1
            for i in range(from_i, to_i + 1):
                if i < len(big_nums):
                    product *= big_nums[i]
                    product %= mod_i
                else:
                    # If we need to calculate beyond precomputed big_nums
                    # We can calculate the powerful array on the fly
                    for power in powerful_array(i):
                        product *= power
                        product %= mod_i
            results.append(product)
        
        return results