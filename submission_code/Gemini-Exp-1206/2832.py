class Solution:
    def longestEqualSubarray(self, nums: List[int], k: int) -> int:
        n = len(nums)
        max_len = 0
        for i in range(n):
            counts = {}
            for j in range(i, n):
                if nums[j] not in counts:
                    counts[nums[j]] = 0
                counts[nums[j]] += 1
                
                max_freq = 0
                for num in counts:
                    max_freq = max(max_freq, counts[num])
                
                if (j - i + 1) - max_freq <= k:
                    max_len = max(max_len, max_freq)
        return max_len