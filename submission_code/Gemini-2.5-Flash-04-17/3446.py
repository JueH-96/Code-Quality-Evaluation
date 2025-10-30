from typing import List

class Solution:
    def numberOfPairs(self, nums1: List[int], nums2: List[int], k: int) -> int:
        count = 0
        n = len(nums1)
        m = len(nums2)

        # Iterate through each element in nums1
        for i in range(n):
            # Iterate through each element in nums2
            for j in range(m):
                # Calculate the required divisor value
                divisor = nums2[j] * k

                # Check if nums1[i] is divisible by the calculated divisor.
                # Since constraints are 1 <= nums2[j], 1 <= k, the divisor is always >= 1.
                if nums1[i] % divisor == 0:
                    count += 1

        return count