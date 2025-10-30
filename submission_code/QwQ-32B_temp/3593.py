from typing import List
import math

class Solution:
    def maxScore(self, nums: List[int]) -> int:
        def compute_gcd(arr):
            current = arr[0]
            for num in arr[1:]:
                current = math.gcd(current, num)
                if current == 1:
                    break  # Early exit as GCD can't be less than 1
            return current

        def compute_lcm(arr):
            current = arr[0]
            for num in arr[1:]:
                current = (current * num) // math.gcd(current, num)
            return current

        max_score = 0

        # Calculate the score for the original array
        original_gcd = compute_gcd(nums)
        original_lcm = compute_lcm(nums)
        max_score = original_gcd * original_lcm

        # Check each possibility of removing one element
        for i in range(len(nums)):
            new_nums = nums[:i] + nums[i+1:]
            if not new_nums:
                current_score = 0
            else:
                g = compute_gcd(new_nums)
                l = compute_lcm(new_nums)
                current_score = g * l
            if current_score > max_score:
                max_score = current_score

        return max_score