class Solution:
    def maximumPrimeDifference(self, nums: List[int]) -> int:
        # The values in nums are between 1 and 100. We can use a pre-defined set of
        # prime numbers in this range for efficient O(1) lookups.
        primes = {2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71, 73, 79, 83, 89, 97}

        # To maximize the distance |j - i|, we need to find the smallest index i
        # and the largest index j where nums[i] and nums[j] are prime.

        # Find the index of the first prime number by searching from the left.
        # We can break early once the first prime is found.
        # The problem guarantees at least one prime, so a value will be found.
        first_idx = 0
        for i in range(len(nums)):
            if nums[i] in primes:
                first_idx = i
                break
        
        # Find the index of the last prime number by searching from the right.
        last_idx = 0
        for i in range(len(nums) - 1, -1, -1):
            if nums[i] in primes:
                last_idx = i
                break
                
        # The maximum distance is the difference between the last and first indices.
        # If only one prime exists, first_idx will equal last_idx, yielding a
        # correct distance of 0.
        return last_idx - first_idx