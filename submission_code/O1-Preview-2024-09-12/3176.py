class Solution:
    def minimumSum(self, nums: List[int]) -> int:
        min_sum = float('inf')
        n = len(nums)
        for j in range(1, n - 1):
            for i in range(j):
                if nums[i] < nums[j]:
                    for k in range(j + 1, n):
                        if nums[k] < nums[j]:
                            curr_sum = nums[i] + nums[j] + nums[k]
                            if curr_sum < min_sum:
                                min_sum = curr_sum
        return min_sum if min_sum != float('inf') else -1