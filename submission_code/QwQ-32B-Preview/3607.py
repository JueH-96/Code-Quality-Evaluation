from typing import List

class Solution:
    def minOperations(self, nums: List[int]) -> int:
        # Precompute smallest prime factors using Sieve of Eratosthenes
        MAX_NUM = 10**6
        spf = [0] * (MAX_NUM + 1)
        for i in range(2, MAX_NUM + 1):
            if spf[i] == 0:
                spf[i] = i
                for j in range(i * 2, MAX_NUM + 1, i):
                    if spf[j] == 0:
                        spf[j] = i
        
        n = len(nums)
        operations = 0
        prev = 1  # Start with the smallest possible value
        
        for num in nums:
            if num == 1:
                if prev > 1:
                    return -1
                prev = 1
            elif spf[num] == num:  # num is prime
                if num < prev:
                    return -1
                prev = num
            else:  # num is composite
                option1 = num
                option2 = spf[num]
                if option1 >= prev:
                    if option2 >= prev:
                        # Choose the smaller option to allow more flexibility for future elements
                        prev = option2
                        operations += 1
                    else:
                        # Can only choose option1
                        prev = option1
                elif option2 >= prev:
                    # Must choose option2
                    prev = option2
                    operations += 1
                else:
                    # Neither option is feasible
                    return -1
        
        return operations