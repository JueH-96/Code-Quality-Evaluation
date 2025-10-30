class Solution:
    def minSizeSubarray(self, nums: List[int], target: int) -> int:
        min_len = float('inf')
        n = len(nums)
        for i in range(2 * n):
            current_sum = 0
            for j in range(i, 2 * n):
                current_sum += nums[j % n]
                if current_sum == target:
                    min_len = min(min_len, j - i + 1)
                    break
                elif current_sum > target:
                    break
        if min_len == float('inf'):
            return -1
        else:
            return min_len