class Solution:
    def maxFrequencyElements(self, nums: List[int]) -> int:
        # Count the frequency of each number
        frequency = {}
        for num in nums:
            frequency[num] = frequency.get(num, 0) + 1
        
        # Find the maximum frequency
        max_freq = max(frequency.values())
        
        # Count how many elements have the maximum frequency
        total_max_freq = sum(freq for freq in frequency.values() if freq == max_freq)
        
        return total_max_freq