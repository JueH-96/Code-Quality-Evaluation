from typing import List

class Solution:
    def maximumPrimeDifference(self, nums: List[int]) -> int:
        # Sieve of Eratosthenes up to the maximum possible value (100)
        max_val = 100
        is_prime = [True] * (max_val + 1)
        is_prime[0] = is_prime[1] = False
        for i in range(2, int(max_val**0.5) + 1):
            if is_prime[i]:
                for j in range(i * i, max_val + 1, i):
                    is_prime[j] = False

        first_prime_index = -1
        last_prime_index = -1

        # Find the first and last occurrence of a prime in nums
        for idx, val in enumerate(nums):
            if is_prime[val]:
                if first_prime_index == -1:
                    first_prime_index = idx
                last_prime_index = idx

        # If there's only one prime, or none (though the problem guarantees ≥1 prime),
        # this will correctly return 0.
        return last_prime_index - first_prime_index