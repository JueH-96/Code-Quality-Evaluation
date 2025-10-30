class Solution:
    def minimumSum(self, nums: List[int]) -> int:
        n = len(nums)
        min_sum = float('inf')
        found = False
        
        for j in range(1, n - 1):
            for i in range(j):
                for k in range(j + 1, n):
                    if nums[i] < nums[j] and nums[k] < nums[j]:
                        found = True
                        current_sum = nums[i] + nums[j] + nums[k]
                        min_sum = min(min_sum, current_sum)
        
        return min_sum if found else -1