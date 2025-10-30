class Solution:
    def minimumSum(self, nums: List[int]) -> int:
        n = len(nums)
        min_sum = float('inf')
        
        # Triple nested loop to check all possible triplets (i, j, k)
        # such that i < j < k, nums[i] < nums[j], and nums[k] < nums[j].
        for j in range(1, n - 1):
            for i in range(j):
                if nums[i] < nums[j]:
                    for k in range(j + 1, n):
                        if nums[k] < nums[j]:
                            current_sum = nums[i] + nums[j] + nums[k]
                            if current_sum < min_sum:
                                min_sum = current_sum
        
        return min_sum if min_sum != float('inf') else -1