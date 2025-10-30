class Solution:
    def maximumPrimeDifference(self, nums: List[int]) -> int:
        primes = {2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71, 73, 79, 83, 89, 97}
        first_occurrence = None
        last_occurrence = None
        
        for i, num in enumerate(nums):
            if num in primes:
                if first_occurrence is None:
                    first_occurrence = i
                last_occurrence = i
        
        return last_occurrence - first_occurrence