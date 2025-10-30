from typing import List

class Solution:
    def minOperations(self, nums: List[int]) -> int:
        n = len(nums)
        if n == 1:
            return 0
        
        operations = 0
        for i in range(n-1, 0, -1):
            if nums[i] < nums[i-1]:
                # Find the greatest proper divisor of nums[i-1] that is <= nums[i]
                # The greatest proper divisor of x is x / smallest_prime_factor(x)
                # So, we need to find the smallest prime factor of nums[i-1]
                x = nums[i-1]
                if x == 1:
                    return -1
                # Find the smallest prime factor of x
                smallest_prime = x
                for p in range(2, int(x**0.5) + 1):
                    if x % p == 0:
                        smallest_prime = p
                        break
                # The greatest proper divisor is x / smallest_prime
                greatest_divisor = x // smallest_prime
                if greatest_divisor > nums[i]:
                    return -1
                nums[i-1] = greatest_divisor
                operations += 1
        
        # Check if the array is non-decreasing
        for i in range(1, n):
            if nums[i] < nums[i-1]:
                return -1
        return operations