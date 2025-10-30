class Solution:
    def maximumPrimeDifference(self, nums: List[int]) -> int:
        # Precompute prime numbers up to 100.
        # A simple approach: we'll list them directly 
        # or check primality by a small function or known set.
        prime_set = set([
            2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 
            31, 37, 41, 43, 47, 53, 59, 61, 67, 
            71, 73, 79, 83, 89, 97
        ])
        
        # Collect indices of prime numbers
        prime_indices = []
        for i, num in enumerate(nums):
            if num in prime_set:
                prime_indices.append(i)
        
        # Since at least one prime is guaranteed to exist,
        # we can safely compute distance
        return prime_indices[-1] - prime_indices[0]  # max distance = last - first