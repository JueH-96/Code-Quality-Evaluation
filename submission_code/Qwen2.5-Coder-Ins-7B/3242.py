class Solution:
    def maxFrequencyElements(self, nums: List[int]) -> int:
        frequency = {}
        for num in nums:
            if num in frequency:
                frequency[num] += 1
            else:
                frequency[num] = 1
        
        max_freq = max(frequency.values())
        total_count = sum(count for count in frequency.values() if count == max_freq)
        
        return total_count