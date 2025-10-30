from typing import List

class Solution:
    def hasIncreasingSubarrays(self, nums: List[int], k: int) -> bool:
        n = len(nums)
        # We need two adjacent subarrays of length k, so start index a goes from 0 to n - 2*k
        for a in range(0, n - 2*k + 1):
            # Check first subarray nums[a..a+k-1]
            if all(nums[i] < nums[i+1] for i in range(a, a + k - 1)):
                # Check second subarray nums[a+k..a+2*k-1]
                b = a + k
                if all(nums[i] < nums[i+1] for i in range(b, b + k - 1)):
                    return True
        return False