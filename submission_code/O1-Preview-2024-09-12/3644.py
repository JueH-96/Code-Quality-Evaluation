class Solution:
    def minimumSumSubarray(self, nums: List[int], l: int, r: int) -> int:
        n = len(nums)
        prefix_sums = [0]*(n+1)
        for i in range(n):
            prefix_sums[i+1] = prefix_sums[i] + nums[i]
        min_sum = float('inf')
        for i in range(n):
            for length in range(l, r+1):
                j = i + length
                if j > n:
                    break
                current_sum = prefix_sums[j] - prefix_sums[i]
                if current_sum > 0 and current_sum < min_sum:
                    min_sum = current_sum
        return min_sum if min_sum != float('inf') else -1