class Solution:
    def minimumIndex(self, nums: List[int]) -> int:
        from collections import Counter

        # Find the dominant element
        count = Counter(nums)
        dominant_element = max(count, key=count.get)
        total_count = count[dominant_element]

        left_count = 0
        for i in range(len(nums) - 1):
            if nums[i] == dominant_element:
                left_count += 1
            right_count = total_count - left_count
            if left_count * 2 > (i + 1) and right_count * 2 > (len(nums) - i - 1):
                return i
        return -1