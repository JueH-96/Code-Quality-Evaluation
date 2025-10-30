class Solution:
    def maxFrequencyElements(self, nums: List[int]) -> int:
        # Count frequency of each element
        freq_count = {}
        for num in nums:
            freq_count[num] = freq_count.get(num, 0) + 1
        
        # Find the maximum frequency
        max_freq = max(freq_count.values())
        
        # Count total elements that have maximum frequency
        total = 0
        for freq in freq_count.values():
            if freq == max_freq:
                total += freq
        
        return total