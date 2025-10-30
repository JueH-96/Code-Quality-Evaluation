from typing import List
from itertools import accumulate
from bisect import bisect_right

class Solution:
    def findProductsOfElements(self, queries: List[List[int]]) -> List[int]:
        def powerful_array(x):
            result = []
            while x:
                power = x & -x
                result.append(power)
                x -= power
            return result
        
        # Precompute the big_nums array up to a reasonable limit
        big_nums = []
        i = 1
        while len(big_nums) < 10**6:  # Arbitrary large limit to ensure we cover the range
            big_nums.extend(powerful_array(i))
            i += 1
        
        # Precompute the prefix products modulo each possible mod value
        prefix_products = {}
        for mod in range(1, 10**5 + 1):
            prefix_products[mod] = list(accumulate(big_nums, lambda x, y: (x * y) % mod, initial=1))
        
        answer = []
        for from_i, to_i, mod_i in queries:
            # Calculate the product of big_nums[from_i:to_i+1] % mod_i
            product = (prefix_products[mod_i][to_i + 1] * pow(prefix_products[mod_i][from_i], -1, mod_i)) % mod_i
            answer.append(product)
        
        return answer