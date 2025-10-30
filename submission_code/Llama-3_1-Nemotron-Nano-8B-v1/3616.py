from typing import List

class Solution:
    def countValidSelections(self, nums: List[int]) -> int:
        n = len(nums)
        count = 0
        # Collect all starting positions where nums[i] is 0
        starts = [i for i in range(n) if nums[i] == 0]
        
        for start in starts:
            for direction in (-1, 1):  # -1 for left, 1 for right
                # Create a copy of the original array to modify
                temp = nums.copy()
                curr = start
                d = direction
                while 0 <= curr < n:
                    if temp[curr] == 0:
                        curr += d
                    else:
                        temp[curr] -= 1
                        d *= -1  # Reverse direction
                        curr += d
                # Check if all elements are zero after the process ends
                if all(x == 0 for x in temp):
                    count += 1
        return count