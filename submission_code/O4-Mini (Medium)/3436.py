from typing import List

class Solution:
    def minimumDifference(self, nums: List[int], k: int) -> int:
        # We maintain at each index the set of all possible OR-values of subarrays ending at that index.
        # It's known that the number of distinct OR-values ending at any index is O(log(max_num)) ~ 32.
        prev = set()
        ans = float('inf')
        
        for num in nums:
            # Start a new subarray with just `num`, and extend all previous subarrays by OR-ing with `num`.
            cur = {num}
            for v in prev:
                cur.add(v | num)
            
            # Update the answer with the best absolute difference for any OR-value ending here.
            for v in cur:
                diff = abs(k - v)
                if diff < ans:
                    ans = diff
                    if ans == 0:
                        return 0  # can't do better than zero
            
            # Move to next position
            prev = cur
        
        return ans