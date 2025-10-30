class Solution:
    def maxFrequencyElements(self, nums: List[int]) -> int:
        # Count frequency of each element
        freq = {}
        for num in nums:
            freq[num] = freq.get(num, 0) + 1
        
        # Find the maximum frequency
        max_freq = max(freq.values())
        
        # Sum the frequencies of elements with maximum frequency
        return sum(count for count in freq.values() if count == max_freq)