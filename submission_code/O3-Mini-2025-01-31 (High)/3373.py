from typing import List

class Solution:
    def maximumPrimeDifference(self, nums: List[int]) -> int:
        # Precompute the set of prime numbers between 1 and 100
        prime_set = {2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 
                     53, 59, 61, 67, 71, 73, 79, 83, 89, 97}
        
        first_index = None
        last_index = None
        
        # Iterate over the array to record the first and last indices of prime numbers.
        for i, num in enumerate(nums):
            if num in prime_set:
                if first_index is None:
                    first_index = i  # mark the first occurrence of a prime.
                last_index = i  # update the last occurrence each time.
                
        # The maximal distance is the difference between the last and first prime indices.
        return last_index - first_index