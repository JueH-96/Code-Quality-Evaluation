class Solution:
    def maximumOr(self, nums: List[int], k: int) -> int:
        n = len(nums)
        if n == 1:
            # If there's only one element, just shift it by k
            return nums[0] << k
        
        # Build prefix OR array
        prefix = [0] * n
        prefix[0] = nums[0]
        for i in range(1, n):
            prefix[i] = prefix[i - 1] | nums[i]
        
        # Build suffix OR array
        suffix = [0] * n
        suffix[n - 1] = nums[n - 1]
        for i in range(n - 2, -1, -1):
            suffix[i] = suffix[i + 1] | nums[i]
        
        # Try applying all k doublings to each element in turn
        max_or = 0
        for i in range(n):
            if i == 0:
                left_part = 0
            else:
                left_part = prefix[i - 1]
            
            if i == n - 1:
                right_part = 0
            else:
                right_part = suffix[i + 1]
            
            # The OR if we shift nums[i] left by k
            candidate = left_part | right_part | (nums[i] << k)
            max_or = max(max_or, candidate)
        
        return max_or