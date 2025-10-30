class Solution:
    def maxFrequency(self, nums: List[int], k: int, numOperations: int) -> int:
        nums.sort()
        n = len(nums)
        left, right = 0, 0
        total = 0
        max_freq = 0

        while right < n:
            total += nums[right]

            while total + k < (right - left + 1) * nums[right]:
                total -= nums[left]
                left += 1

            max_freq = max(max_freq, right - left + 1)

            if max_freq + numOperations >= n:
                return n

            right += 1

        return max_freq