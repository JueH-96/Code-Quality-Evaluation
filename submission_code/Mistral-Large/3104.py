from typing import List

class Solution:
    def countWays(self, nums: List[int]) -> int:
        n = len(nums)
        count = 0

        # Check if selecting no students makes everyone happy
        if all(num > 0 for num in nums):
            count += 1

        # Check if selecting all students makes everyone happy
        if all(num < n for num in nums):
            count += 1

        # Check for selecting a subset of students
        for k in range(1, n):
            happy = True
            for num in nums:
                if not ((num < k) or (num > k)):
                    happy = False
                    break
            if happy:
                count += 1

        return count