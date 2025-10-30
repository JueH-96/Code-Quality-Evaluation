from typing import List

class Solution:
    def hasIncreasingSubarrays(self, nums: List[int], k: int) -> bool:
        n = len(nums)
        # If there aren't enough elements for two subarrays of length k, return False
        if n < 2 * k:
            return False
        
        # Try each possible starting index a for the first subarray
        # The second subarray will start at b = a + k
        for a in range(n - 2 * k + 1):
            # Check first subarray nums[a..a+k-1] is strictly increasing
            ok = True
            for i in range(a + 1, a + k):
                if nums[i] <= nums[i - 1]:
                    ok = False
                    break
            if not ok:
                continue
            
            # Check second subarray nums[a+k..a+2k-1] is strictly increasing
            b = a + k
            for i in range(b + 1, b + k):
                if nums[i] <= nums[i - 1]:
                    ok = False
                    break
            
            if ok:
                return True
        
        return False