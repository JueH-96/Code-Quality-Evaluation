class Solution:
    def maximumPrimeDifference(self, nums: List[int]) -> int:
        # Set of all primes from 1 to 100
        prime_set = {2, 3, 5, 7, 11, 13, 17, 19, 23, 29,
                     31, 37, 41, 43, 47, 53, 59, 61, 67, 71,
                     73, 79, 83, 89, 97}
        
        # Collect indices where nums[i] is prime
        prime_indices = [i for i, val in enumerate(nums) if val in prime_set]
        
        # The maximum distance is between the first and last prime index
        return prime_indices[-1] - prime_indices[0]