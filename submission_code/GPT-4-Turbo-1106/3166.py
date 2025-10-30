from collections import Counter

class Solution:
    def minGroupsForValidAssignment(self, nums: list[int]) -> int:
        # Count the frequency of each number in nums
        freq = Counter(nums)
        
        # The maximum frequency determines the minimum number of groups needed
        max_freq = max(freq.values())
        
        # Calculate the number of elements with the maximum frequency
        max_freq_count = sum(1 for value in freq.values() if value == max_freq)
        
        # If the number of elements with the maximum frequency is greater than the maximum frequency,
        # then we need one extra group to ensure the difference between group sizes does not exceed 1
        if max_freq_count > max_freq:
            return max_freq + 1
        
        # Otherwise, the maximum frequency is the minimum number of groups needed
        return max_freq