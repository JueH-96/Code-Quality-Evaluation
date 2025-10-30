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

        prime_indices = [i for i, num in enumerate(nums) if is_prime(num)]
        if not prime_indices:
            return 0

        return max(prime_indices) - min(prime_indices)