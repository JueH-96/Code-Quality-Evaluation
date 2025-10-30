class Solution:
    def maxFrequencyElements(self, nums: List[int]) -> int:
        freq = {}
        for num in nums:
            if num not in freq:
                freq[num] = 1
            else:
                freq[num] += 1
        max_freq = max(freq.values())
        total_max_freq_elements = sum(1 for num, count in freq.items() if count == max_freq)
        return total_max_freq_elements * max_freq