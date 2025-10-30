from typing import List

class Solution:
    def minimumTime(self, nums1: List[int], nums2: List[int], x: int) -> int:
        n = len(nums1)
        total_sum = sum(nums1)
        if total_sum > x:
            # If the initial sum is already greater than x, we need to check if we can make it smaller
            if all(num2 == 0 for num2 in nums2):
                # If all nums2 are 0, we can't make the sum smaller
                return -1
            else:
                # Try to make the sum smaller by setting the largest nums1 to 0
                max_index = nums1.index(max(nums1))
                nums1[max_index] = 0
                total_sum -= max(nums1)
                if total_sum <= x:
                    return 1
                else:
                    # If we still can't make the sum smaller, return -1
                    return -1
        
        time = 0
        while total_sum > x:
            # Find the index with the largest nums2
            max_index = nums2.index(max(nums2))
            # Increment all nums1 by their corresponding nums2
            for i in range(n):
                nums1[i] += nums2[i]
            # Set the nums1 with the largest nums2 to 0
            nums1[max_index] = 0
            # Update the total sum
            total_sum = sum(nums1)
            # Increment the time
            time += 1
            # If the total sum is still greater than x and all nums2 are 0, return -1
            if total_sum > x and all(num2 == 0 for num2 in nums2):
                return -1
        return time