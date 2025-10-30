from typing import List

class Solution:
    def maxLength(self, nums: List[int]) -> int:
        primes = [2, 3, 5, 7]
        count = {p: 0 for p in primes}
        left = 0
        max_length = 0
        n = len(nums)
        for right in range(n):
            num_right = nums[right]
            if num_right % 2 == 0:
                count[2] += 1
            if num_right % 3 == 0:
                count[3] += 1
            if num_right % 5 == 0:
                count[5] += 1
            if num_right % 7 == 0:
                count[7] += 1
            while max(count.values()) > 1 and left <= right:
                num_left = nums[left]
                if num_left % 2 == 0:
                    count[2] -= 1
                if num_left % 3 == 0:
                    count[3] -= 1
                if num_left % 5 == 0:
                    count[5] -= 1
                if num_left % 7 == 0:
                    count[7] -= 1
                left += 1
            current_length = right - left + 1
            max_length = max(max_length, current_length)
        return max(max_length, 2)