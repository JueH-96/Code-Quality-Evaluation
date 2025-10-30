class Solution:
    def minimumIndex(self, nums: List[int]) -> int:
        from collections import defaultdict

        # Find the dominant element
        freq = defaultdict(int)
        dominant = None
        for num in nums:
            freq[num] += 1
            if freq[num] * 2 > len(nums):
                dominant = num
                break

        # Now, find the minimum index i where both left and right subarrays have dominant as the dominant element
        left_count = 0
        for i in range(len(nums)):
            if nums[i] == dominant:
                left_count += 1
            # Check if the left subarray has dominant as the dominant element
            if left_count * 2 > (i + 1):
                # Now, check the right subarray
                right_count = freq[dominant] - left_count
                if right_count * 2 > (len(nums) - (i + 1)):
                    return i
        return -1