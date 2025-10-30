from typing import List
from itertools import combinations

class Solution:
    def maxProduct(self, nums: List[int], k: int, limit: int) -> int:
        n = len(nums)
        max_product = -1
        
        # Iterate over all possible non-empty subsequences
        for r in range(1, n + 1):
            for subseq in combinations(nums, r):
                # Calculate alternating sum
                alt_sum = sum(subseq[i] if i % 2 == 0 else -subseq[i] for i in range(len(subseq)))
                
                # Check if the alternating sum is equal to k
                if alt_sum == k:
                    # Calculate product
                    product = 1
                    for num in subseq:
                        product *= num
                        # If product exceeds limit, break early
                        if product > limit:
                            break
                    
                    # Update max_product if product is within limit
                    if product <= limit:
                        max_product = max(max_product, product)
        
        return max_product