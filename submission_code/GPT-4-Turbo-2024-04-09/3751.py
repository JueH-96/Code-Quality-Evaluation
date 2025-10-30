class Solution:
    def maxFrequency(self, nums: List[int], k: int) -> int:
        from collections import defaultdict
        
        # Frequency of each number before any operation
        freq = defaultdict(int)
        for num in nums:
            freq[num] += 1
        
        # If k is already the most frequent without any operation
        if freq[k] == len(nums):
            return freq[k]
        
        # Maximum frequency of k after operation
        max_freq = freq[k]
        
        # Calculate the difference needed to make any element into k
        diff_count = defaultdict(int)
        for num in nums:
            diff_count[k - num] += 1
        
        # Iterate over possible differences and calculate the potential maximum frequency
        for diff, count in diff_count.items():
            if diff == 0:
                # This means no change needed, already handled in initial max_freq
                continue
            # Calculate how many times we can achieve k by adding 'diff' to some elements
            potential_freq = freq[k] + count
            max_freq = max(max_freq, potential_freq)
        
        return max_freq