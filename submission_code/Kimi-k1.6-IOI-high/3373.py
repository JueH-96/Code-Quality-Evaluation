class Solution:
    def maximumPrimeDifference(self, nums: List[int]) -> int:
        primes = {2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71, 73, 79, 83, 89, 97}
        prime_indices = []
        for idx, num in enumerate(nums):
            if num in primes:
                prime_indices.append(idx)
        return prime_indices[-1] - prime_indices[0] if len(prime_indices) > 1 else 0