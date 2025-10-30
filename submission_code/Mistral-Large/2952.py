from typing import List

class Solution:
    def minimumTime(self, nums1: List[int], nums2: List[int], x: int) -> int:
        n = len(nums1)
        # Create a list of tuples (nums1[i] - nums2[i], nums2[i])
        diff = [(nums1[i] - nums2[i], nums2[i]) for i in range(n)]
        # Sort the list based on nums1[i] - nums2[i]
        diff.sort()

        # Initialize variables
        current_sum = sum(nums1)
        time = 0

        # Iterate over the sorted list
        for i in range(n):
            # If the current sum is already less than or equal to x, return the time
            if current_sum <= x:
                return time
            # Increment time
            time += 1
            # Update the current sum by subtracting nums2[i] and adding nums1[i] - nums2[i]
            current_sum += diff[i][0] - diff[i][1]

        # If we exit the loop and the current sum is still greater than x, return -1
        return -1 if current_sum > x else time