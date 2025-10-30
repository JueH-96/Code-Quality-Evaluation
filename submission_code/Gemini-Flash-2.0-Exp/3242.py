class Solution:
    def maxFrequencyElements(self, nums: List[int]) -> int:
        counts = {}
        for num in nums:
            counts[num] = counts.get(num, 0) + 1
        
        max_freq = 0
        for num in counts:
            max_freq = max(max_freq, counts[num])
        
        ans = 0
        for num in nums:
            if counts[num] == max_freq:
                ans += 1
        
        return ans