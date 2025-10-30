class Solution:
    def maxFrequencyElements(self, nums: List[int]) -> int:
        # Count the frequency of each element in the array
        freq = {}
        for num in nums:
            freq[num] = freq.get(num, 0) + 1
        
        # Find the maximum frequency
        max_freq = max(freq.values())
        
        # Count the number of elements with the maximum frequency
        return sum(1 for count in freq.values() if count == max_freq)