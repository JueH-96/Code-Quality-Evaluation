class Solution:
    def minimumSumSubarray(self, nums: List[int], l: int, r: int) -> int:
        min_sum = float('inf')
        n = len(nums)
        for i in range(n):
            current_sum = 0
            for j in range(i, n):
                current_sum += nums[j]
                window_length = j - i + 1
                if window_length >= l and window_length <= r:
                    if current_sum > 0:
                        if current_sum < min_sum:
                            min_sum = current_sum
        return min_sum if min_sum != float('inf') else -1