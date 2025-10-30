class Solution:
    def minimumSum(self, nums: List[int]) -> int:
        n = len(nums)
        min_sum = float('inf')
        found = False
        
        # triple nested loop over indices i, j, k
        for j in range(1, n - 1):
            for i in range(j):
                if nums[i] >= nums[j]:
                    continue
                for k in range(j + 1, n):
                    if nums[k] >= nums[j]:
                        continue
                    curr_sum = nums[i] + nums[j] + nums[k]
                    if curr_sum < min_sum:
                        min_sum = curr_sum
                        found = True
        
        return min_sum if found else -1