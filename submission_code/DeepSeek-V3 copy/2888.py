class Solution:
    def minimumIndex(self, nums: List[int]) -> int:
        from collections import defaultdict

        # Find the dominant element in the entire array
        freq = defaultdict(int)
        for num in nums:
            freq[num] += 1
        dominant = None
        for num in freq:
            if freq[num] * 2 > len(nums):
                dominant = num
                break

        # Now, iterate through the array to find the minimum index where the split is valid
        left_count = 0
        for i in range(len(nums)):
            if nums[i] == dominant:
                left_count += 1
            # Check if the left subarray has dominant as the dominant element
            if left_count * 2 > (i + 1):
                # Check if the right subarray also has dominant as the dominant element
                right_count = freq[dominant] - left_count
                if right_count * 2 > (len(nums) - (i + 1)):
                    return i
        return -1