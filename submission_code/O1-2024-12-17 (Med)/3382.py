from typing import List
import bisect

class Solution:
    def numberOfSubarrays(self, nums: List[int]) -> int:
        n = len(nums)
        
        # Step 1: Compute the array next_greater where next_greater[i] is the
        # smallest index k > i such that nums[k] > nums[i], or n if there is none.
        next_greater = [n] * n
        stack = []
        # We'll process from right to left.
        for i in range(n - 1, -1, -1):
            while stack and nums[stack[-1]] <= nums[i]:
                stack.pop()
            next_greater[i] = stack[-1] if stack else n
            stack.append(i)
        
        # Step 2: Build a dictionary "pos" mapping each value v to 
        # a sorted list of indices where nums[index] == v.
        pos = {}
        for i, v in enumerate(nums):
            if v not in pos:
                pos[v] = []
            pos[v].append(i)
        
        # Step 3: For each index i, let v = nums[i]. Valid subarrays start at i
        # and end at any j in [i, next_greater[i] - 1] where nums[j] == v.
        # Use binary search on pos[v] to count such j's.
        answer = 0
        for i, v in enumerate(nums):
            L = i
            R = next_greater[i] - 1
            # Find how many indices of v lie in the range [L, R].
            left_idx = bisect.bisect_left(pos[v], L)
            right_idx = bisect.bisect_right(pos[v], R)
            answer += (right_idx - left_idx)
        
        return answer