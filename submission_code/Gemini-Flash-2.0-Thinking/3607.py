from typing import List

class Solution:
    def minOperations(self, nums: List[int]) -> int:
        n = len(nums)
        operations = 0

        def get_greatest_proper_divisor(num):
            if num <= 1:
                return 1
            for i in range(2, int(num**0.5) + 1):
                if num % i == 0:
                    return num // i
            return 1

        def reduce_number(num):
            if num <= 1:
                return 0, num
            gcd = get_greatest_proper_divisor(num)
            if gcd == 1:
                return 0, num
            return 1, num // gcd

        i = 0
        while i < n - 1:
            if nums[i] > nums[i+1]:
                if nums[i] == 1:
                    return -1

                gcd = get_greatest_proper_divisor(nums[i])
                if gcd == 1:
                    return -1  # Cannot reduce further

                nums[i] //= gcd
                operations += 1
            else:
                i += 1

        # Verify if the array is non-decreasing
        for i in range(n - 1):
            if nums[i] > nums[i+1]:
                return -1

        return operations