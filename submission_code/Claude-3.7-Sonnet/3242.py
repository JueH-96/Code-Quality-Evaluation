class Solution:
    def maxFrequencyElements(self, nums: List[int]) -> int:
        from collections import Counter
        
        # Count the frequency of each element
        counter = Counter(nums)
        
        # Find the maximum frequency
        max_freq = max(counter.values())
        
        # Count how many elements have this maximum frequency
        count_elements_with_max_freq = sum(1 for freq in counter.values() if freq == max_freq)
        
        # Return total frequencies (max_freq × number of elements with max_freq)
        return max_freq * count_elements_with_max_freq