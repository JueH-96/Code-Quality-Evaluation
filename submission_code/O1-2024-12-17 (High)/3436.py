from typing import List

class Solution:
    def minimumDifference(self, nums: List[int], k: int) -> int:
        dp = set()  # will hold all distinct ORs of subarrays ending at the current index
        best = float('inf')

        for x in nums:
            new_dp = {x}
            # Extend all previous subarrays by OR'ing with the new element
            for val in dp:
                new_dp.add(val | x)
            dp = new_dp

            # Check each OR value for closeness to k
            for val in dp:
                diff = abs(k - val)
                if diff < best:
                    best = diff
                    # If we ever hit 0, we cannot do better
                    if best == 0:
                        return 0
        
        return best