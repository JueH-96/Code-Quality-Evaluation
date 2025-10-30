from typing import List
from collections import Counter
import math

class Solution:
    def minGroupsForValidAssignment(self, nums: List[int]) -> int:
        # Count the frequency of each number in nums
        freq = Counter(nums)

        # Function to check if we can form k groups with the given frequencies
        def canFormGroups(k: int) -> bool:
            for count in freq.values():
                if count > k:
                    groupsNeeded = math.ceil(count / k)
                    remaining = count % k
                    if remaining > (groupsNeeded * (k + 1) - count):
                        return False
            return True

        # Binary search for the minimum number of groups
        left, right = 1, max(freq.values())
        while left < right:
            mid = (left + right) // 2
            if canFormGroups(mid):
                right = mid
            else:
                left = mid + 1

        return left