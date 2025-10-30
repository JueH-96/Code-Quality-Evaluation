class Solution:
    def minimumSum(self, nums: List[int]) -> int:
        min_sum = float('inf')
        n = len(nums)
        for j in range(1, n - 1):
            left = nums[:j]
            right = nums[j+1:]
            left_min = min(left)
            right_min = min(right)
            if left_min < nums[j] and right_min < nums[j]:
                current_sum = left_min + nums[j] + right_min
                if current_sum < min_sum:
                    min_sum = current_sum
        return min_sum if min_sum != float('inf') else -1