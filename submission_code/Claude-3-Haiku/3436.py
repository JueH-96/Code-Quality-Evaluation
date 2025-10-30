class Solution:
    def minimumDifference(self, nums: List[int], k: int) -> int:
        n = len(nums)
        if n < k:
            return 0
        
        nums.sort()
        min_diff = float('inf')
        
        for i in range(n - k + 1):
            min_diff = min(min_diff, abs(k - (nums[i] | nums[i+1] | ... | nums[i+k-1])))
        
        return min_diff