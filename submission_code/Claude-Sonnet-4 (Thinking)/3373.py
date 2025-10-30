class Solution:
    def maximumPrimeDifference(self, nums: List[int]) -> int:
        # Precomputed prime numbers up to 100
        primes = {2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71, 73, 79, 83, 89, 97}
        
        # Find all indices where prime numbers occur
        prime_indices = [i for i, num in enumerate(nums) if num in primes]
        
        # Return the distance between first and last prime indices
        return prime_indices[-1] - prime_indices[0]