class Solution:
    def longestMonotonicSubarray(self, nums: List[int]) -> int:
        if len(nums) < 2:
            return len(nums)
        
        increasing, decreasing = [1], [1]
        for i in range(1, len(nums)):
            if nums[i] > nums[i-1]:
                increasing.append(increasing[-1] + 1)
                decreasing.append(1)
            elif nums[i] < nums[i-1]:
                decreasing.append(decreasing[-1] + 1)
                increasing.append(1)
            else:
                decreasing.append(1)
                increasing.append(1)
        
        return max(max(increasing), max(decreasing))