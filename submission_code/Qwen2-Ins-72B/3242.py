class Solution:
    def maxFrequencyElements(self, nums: List[int]) -> int:
        # Create a dictionary to store the frequency of each element
        freq = {}
        for num in nums:
            if num in freq:
                freq[num] += 1
            else:
                freq[num] = 1
        
        # Find the maximum frequency
        max_freq = max(freq.values())
        
        # Sum the frequencies of elements with the maximum frequency
        total_freq = sum(f for f in freq.values() if f == max_freq)
        
        return total_freq