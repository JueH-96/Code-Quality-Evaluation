class Solution:
    def maxFrequencyScore(self, nums: List[int], k: int) -> int:
        nums.sort()
        n = len(nums)
        prefix_sum = [0] * (n + 1)

        for i in range(n):
            prefix_sum[i + 1] = prefix_sum[i] + nums[i]

        def cost(l, r):
            mid = (l + r) // 2
            target = nums[mid]
            left_cost = target * (mid - l + 1) - (prefix_sum[mid + 1] - prefix_sum[l])
            right_cost = (prefix_sum[r + 1] - prefix_sum[mid]) - target * (r - mid)
            return left_cost + right_cost

        left = 0
        max_freq = 1

        for right in range(n):
            while cost(left, right) > k:
                left += 1
            max_freq = max(max_freq, right - left + 1)

        return max_freq