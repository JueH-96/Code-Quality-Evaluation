from typing import List

class Solution:
    def minOperations(self, nums: List[int], k: int) -> int:
        # Since we can only decrease numbers, if any number is smaller than k,
        # we can never bring it up to k.
        if min(nums) < k:
            return -1
        
        # Observation:
        # The only meaningful operations are those that reduce numbers 
        # that are strictly greater than k.
        # At every step, if you choose a valid integer h, then all numbers greater than h
        # are identical. This forces us to reduce the current maximum in one step
        # to the next smaller value (or eventually to k when it's valid).
        # Thus, the minimum number of operations is exactly the number of distinct values in nums that are greater than k.
        
        return len({x for x in nums if x > k})

# Example test runs:
if __name__ == "__main__":
    sol = Solution()
    # Example 1
    print(sol.minOperations([5,2,5,4,5], 2))  # Expected output: 2
    # Example 2
    print(sol.minOperations([2,1,2], 2))      # Expected output: -1
    # Example 3
    print(sol.minOperations([9,7,5,3], 1))      # Expected output: 4