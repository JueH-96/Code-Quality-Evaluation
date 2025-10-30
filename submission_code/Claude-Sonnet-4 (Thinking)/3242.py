class Solution:
    def maxFrequencyElements(self, nums: List[int]) -> int:
        # Count frequency of each element
        freq = {}
        for num in nums:
            freq[num] = freq.get(num, 0) + 1
        
        # Find the maximum frequency
        max_freq = max(freq.values())
        
        # Count total elements with maximum frequency
        result = 0
        for frequency in freq.values():
            if frequency == max_freq:
                result += frequency
        
        return result