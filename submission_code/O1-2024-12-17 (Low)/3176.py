class Solution:
    def minimumSum(self, nums: List[int]) -> int:
        n = len(nums)
        min_sum = float('inf')
        # We can afford O(n^3) for n <= 50
        for j in range(1, n-1):
            for i in range(j):
                if nums[i] < nums[j]:
                    for k in range(j+1, n):
                        if nums[k] < nums[j]:
                            current_sum = nums[i] + nums[j] + nums[k]
                            min_sum = min(min_sum, current_sum)
        return min_sum if min_sum != float('inf') else -1