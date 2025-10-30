class Solution:
    def maximumPrimeDifference(self, nums: List[int]) -> int:
        primes = {2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71, 73, 79, 83, 89, 97}
        first_prime_idx = -1
        last_prime_idx = -1
        for i, num in enumerate(nums):
            if num in primes:
                if first_prime_idx == -1:
                    first_prime_idx = i
                last_prime_idx = i
        return last_prime_idx - first_prime_idx