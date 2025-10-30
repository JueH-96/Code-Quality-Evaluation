from typing import List

class Solution:
    def numberOfSubarrays(self, nums: List[int]) -> int:
        n = len(nums)
        stack = []
        total_count = 0

        for i in range(n):
            while stack and nums[i] > nums[stack[-1]]:
                left = stack.pop()
                if stack:
                    segment_left = stack[-1] + 1
                else:
                    segment_left = 0
                segment_right = i - 1
                c = self.count_occurrences(nums, segment_left, segment_right, nums[left])
                total_count += c * (c + 1) // 2
            stack.append(i)

        while stack:
            left = stack.pop()
            if stack:
                segment_left = stack[-1] + 1
            else:
                segment_left = 0
            segment_right = n - 1
            c = self.count_occurrences(nums, segment_left, segment_right, nums[left])
            total_count += c * (c + 1) // 2

        return total_count

    def count_occurrences(self, nums, left, right, X):
        count = 0
        for i in range(left, right + 1):
            if nums[i] == X:
                count += 1
        return count