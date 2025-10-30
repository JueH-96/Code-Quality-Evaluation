class Solution:
    def maxFrequencyElements(self, nums: List[int]) -> int:
        from collections import Counter

        # Count the frequency of each element in the array
        frequency = Counter(nums)

        # Find the maximum frequency
        max_freq = max(frequency.values())

        # Count the number of elements that have the maximum frequency
        count_max_freq = sum(1 for count in frequency.values() if count == max_freq)

        # Return the total number of elements with the maximum frequency
        return count_max_freq * max_freq