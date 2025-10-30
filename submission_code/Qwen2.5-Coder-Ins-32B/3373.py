from typing import List

class Solution:
    def maximumPrimeDifference(self, nums: List[int]) -> int:
        def is_prime(n: int) -> bool:
            if n <= 1:
                return False
            if n <= 3:
                return True
            if n % 2 == 0 or n % 3 == 0:
                return False
            i = 5
            while i * i <= n:
                if n % i == 0 or n % (i + 2) == 0:
                    return False
                i += 6
            return True
        
        first_prime_index = -1
        last_prime_index = -1
        
        for i, num in enumerate(nums):
            if is_prime(num):
                if first_prime_index == -1:
                    first_prime_index = i
                last_prime_index = i
        
        return last_prime_index - first_prime_index