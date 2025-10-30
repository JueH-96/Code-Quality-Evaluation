class Solution:
    def maxFrequency(self, nums: List[int], k: int) -> int:
        from collections import defaultdict
        
        # Calculate the difference of each element from k
        differences = [k - num for num in nums]
        
        # Use a dictionary to count the frequency of each difference
        freq = defaultdict(int)
        
        # Calculate the maximum frequency of any difference
        max_freq = 0
        for diff in differences:
            freq[diff] += 1
            max_freq = max(max_freq, freq[diff])
        
        return max_freq