import math

class Solution:
    def minimumArrayLength(self, nums: List[int]) -> int:
        ones_count = 0
        all_ones = True
        has_greater_than_one = False
        for num in nums:
            if num == 1:
                ones_count += 1
            else:
                all_ones = False
                has_greater_than_one = True
        if ones_count > 0:
            if all_ones:
                return (len(nums) + 1) // 2
            else:
                return 1
        else:
            current_gcd = nums[0]
            for i in range(1, len(nums)):
                current_gcd = math.gcd(current_gcd, nums[i])
            if current_gcd == 1:
                return 1
            else:
                return 2