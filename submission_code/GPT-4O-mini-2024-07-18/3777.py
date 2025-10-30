from typing import List
from itertools import combinations

class Solution:
    def maxProduct(self, nums: List[int], k: int, limit: int) -> int:
        max_product = -1
        
        # Generate all non-empty subsequences
        n = len(nums)
        for r in range(1, n + 1):
            for subseq in combinations(nums, r):
                # Calculate the alternating sum
                alternating_sum = sum(subseq[i] if i % 2 == 0 else -subseq[i] for i in range(len(subseq)))
                
                # Check if the alternating sum matches k
                if alternating_sum == k:
                    # Calculate the product
                    product = 1
                    for num in subseq:
                        product *= num
                    
                    # Check if the product is within the limit and maximize it
                    if product <= limit:
                        max_product = max(max_product, product)
        
        return max_product