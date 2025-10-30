from typing import List
from collections import defaultdict

class Solution:
    def numberOfSubsequences(self, nums: List[int]) -> int:
        product_to_pairs = defaultdict(list)
        n = len(nums)
        
        # Generate all valid pairs (p, r) with r >= p + 2
        for p in range(n - 3):
            for r in range(p + 2, n - 1):
                product = nums[p] * nums[r]
                product_to_pairs[product].append((p, r))
        
        count = 0
        
        # Iterate through each product group
        for product, pairs in product_to_pairs.items():
            # Sort pairs based on the first index to ensure p1 < p2
            pairs.sort()
            m = len(pairs)
            
            # Compare each pair with every other pair in the same product group
            for i in range(m):
                p1, r1 = pairs[i]
                for j in range(i + 1, m):
                    p2, s2 = pairs[j]
                    # Check the index constraints
                    if p2 > p1 + 1 and p2 < r1 - 1 and s2 > r1 + 1:
                        count += 1
        
        return count