from typing import List

class Solution:
    def numberOfSubarrays(self, nums: List[int]) -> int:
        n = len(nums)
        res = 0
        stack = []
        left = [0] * n
        right = [n] * n

        # Find the next greater element to the left
        for i in range(n):
            while stack and nums[stack[-1]] < nums[i]:
                stack.pop()
            left[i] = stack[-1] if stack else -1
            stack.append(i)
        
        stack.clear()

        # Find the next greater element to the right
        for i in range(n-1, -1, -1):
            while stack and nums[stack[-1]] <= nums[i]:
                stack.pop()
            right[i] = stack[-1] if stack else n
            stack.append(i)
        
        # For each element, calculate the number of valid subarrays
        for i in range(n):
            l = left[i]
            r = right[i]
            # Count how many subarrays where nums[i] is the maximum
            # and it is both the first and the last element
            res += (i - l) * (r - i)
        
        return res