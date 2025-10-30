class Solution:
    def maximumPrimeDifference(self, nums: List[int]) -> int:
        # Set of all prime numbers up to 100
        primes = {2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71, 73, 79, 83, 89, 97}
        
        min_prime_index = float('inf')
        max_prime_index = float('-inf')
        
        for i, num in enumerate(nums):
            if num in primes:
                min_prime_index = min(min_prime_index, i)
                max_prime_index = max(max_prime_index, i)
        
        if min_prime_index == max_prime_index:
            return 0
        
        return max_prime_index - min_prime_index