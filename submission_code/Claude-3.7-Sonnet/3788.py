class Solution:
    def maxSum(self, nums: List[int]) -> int:
        max_sum = float('-inf')
        n = len(nums)
        
        for start_idx in range(n):
            seen = set()
            current_sum = 0
            start = start_idx
            
            for end in range(start_idx, n):
                # If we encounter a duplicate, shrink the window from the left
                while nums[end] in seen:
                    seen.remove(nums[start])
                    current_sum -= nums[start]
                    start += 1
                
                seen.add(nums[end])
                current_sum += nums[end]
                max_sum = max(max_sum, current_sum)
        
        return max_sum