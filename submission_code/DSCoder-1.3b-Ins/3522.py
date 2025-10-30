class Solution:
    def resultsArray(self, nums: List[int], k: int) -> List[int]:
        n = len(nums)
        if k == 1:
            return [-1]*n
        
        max_subarray = [-1]*n
        stack = []
        
        for i in range(n):
            while stack and nums[i] > nums[stack[-1]]:
                stack.pop()
            if stack and i - stack[-1] == k - 1:
                max_subarray[i] = nums[stack[-1]]
            if i >= k - 1:
                stack.append(i)
        
        return max_subarray