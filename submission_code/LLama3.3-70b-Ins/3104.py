from typing import List
import itertools

class Solution:
    def countWays(self, nums: List[int]) -> int:
        n = len(nums)
        count = 0
        
        # Generate all possible subsets of students
        for r in range(n + 1):
            for subset in itertools.combinations(range(n), r):
                # Check if all students are happy with the current subset
                happy = True
                for i in range(n):
                    if i in subset:
                        if len(subset) <= nums[i]:
                            happy = False
                            break
                    else:
                        if len(subset) >= nums[i]:
                            happy = False
                            break
                # If all students are happy, increment the count
                if happy:
                    count += 1
        
        return count