from typing import List
from collections import defaultdict

class Solution:
    def numberOfSubsequences(self, nums: List[int]) -> int:
        n = len(nums)
        answer = 0
        
        # We need p < q < r < s with:
        #   q - p > 1  => q >= p+2
        #   r - q > 1  => r >= q+2
        #   s - r > 1  => s >= r+2
        #
        # A valid quadruple (p, q, r, s) must satisfy:
        #   nums[p] * nums[r] == nums[q] * nums[s]
        #
        # Strategy (O(n^3) approach):
        # 1) Fix r (the third index).
        # 2) Incrementally build a dictionary "left" that counts products nums[p] * nums[r]
        #    for all valid p (those with p+2 <= q).
        # 3) For each q (the second index) with q+2 <= r, build a frequency table of
        #    products = nums[q] * nums[s] for s in [r+2..n-1].
        # 4) Combine counts of matching products in "left" with the frequency table,
        #    summing up how many p and s form the same product.
        
        for r in range(4, n):
            # If r+2 goes beyond array, there's no valid s:
            if r + 2 >= n:
                break
            
            # Dictionary to count products nums[p] * nums[r] for all valid p up to current q
            left_products = defaultdict(int)
            
            # pAdd will track how many p's have been added so far.
            # For each q, we need p in [0..q-2].
            pAdd = 0
            
            # q goes from 2 up to r-1, ensuring q+2 <= r.
            for q in range(2, r):
                if q + 2 > r:
                    break
                
                # Add all new p values up to q-2 into left_products
                while pAdd <= q - 2:
                    product_p = nums[pAdd] * nums[r]
                    left_products[product_p] += 1
                    pAdd += 1
                
                # Build frequency table for products = nums[q] * nums[s], s in [r+2..n-1].
                freq_s = defaultdict(int)
                for s in range(r+2, n):
                    product_s = nums[q] * nums[s]
                    freq_s[product_s] += 1
                
                # Count how many matches with left_products
                for prod_val, count_s in freq_s.items():
                    answer += left_products.get(prod_val, 0) * count_s
        
        return answer