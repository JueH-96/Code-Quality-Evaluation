class Solution:
    def maxFrequencyScore(self, nums: List[int], k: int) -> int:
        nums.sort()
        n = len(nums)
        prefix = [0]*n
        prefix[0] = nums[0]
        for i in range(1, n):
            prefix[i] = prefix[i-1] + nums[i]
        
        max_freq = 1
        left = 0
        for right in range(n):
            while left < right:
                mid = (left + right) // 2
                left_sum = prefix[mid -1] - (prefix[left -1] if left > 0 else 0)
                left_cost = nums[mid] * (mid - left) - left_sum
                right_sum = prefix[right] - prefix[mid]
                right_cost = right_sum - nums[mid] * (right - mid)
                total_cost = left_cost + right_cost
                if total_cost <= k:
                    break
                left += 1
            max_freq = max(max_freq, right - left +1)
        return max_freq