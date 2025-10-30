import math

class Solution:
    def maxLength(self, nums: List[int]) -> int:
        max_length = 0
        n = len(nums)
        for i in range(n):
            current_product = 1
            current_gcd = nums[i]
            current_lcm = nums[i]
            for j in range(i, n):
                current_product *= nums[j]
                current_gcd = math.gcd(current_gcd, nums[j])
                current_lcm = (current_lcm * nums[j]) // math.gcd(current_lcm, nums[j])
                if current_product == current_gcd * current_lcm:
                    current_length = j - i + 1
                    if current_length > max_length:
                        max_length = current_length
        return max_length