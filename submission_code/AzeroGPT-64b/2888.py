class Solution:
    def minimumIndex(self, nums: List[int]) -> int:
        count = defaultdict(int)
        for val in nums:
            count[val] += 1
        (dominant, freq), = count.most_common(1)  # get the dominant element and its frequency

        left_freq = 0  # frequency of dominant element in the left subarray
        for i, val in enumerate(nums):
            if val == dominant:
                left_freq += 1
            if left_freq * 2 > i + 1 and (freq - left_freq) * 2 > len(nums) - i - 1:
                return i
        return -1