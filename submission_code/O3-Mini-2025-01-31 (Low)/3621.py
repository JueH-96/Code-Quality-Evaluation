from typing import List

class Solution:
    def minOperations(self, nums: List[int], k: int) -> int:
        # Since we are only allowed to decrease numbers,
        # every number in nums must be at least k.
        for num in nums:
            if num < k:
                return -1
        
        # The operation works as follows:
        # We pick a valid integer h such that all elements > h are equal.
        # When we do such an operation, we reduce one set (or “block”) of numbers.
        # In an optimal strategy, we want to remove one distinct block of numbers in each step.
        #
        # Thus, if we look at the distinct values in nums that are greater than k,
        # each such distinct value will eventually have to be reduced by one operation.
        #
        # For example:
        #  nums = [5,2,5,4,5], k = 2 -> distinct values are {2,4,5}
        #  Values strictly greater than 2 are 4 and 5, i.e. 2 blocks -> answer 2.
        # 
        #  nums = [9,7,5,3], k = 1
        #   distinct = {3,5,7,9} (and note k=1 is not in nums but is the target)
        #   All values are > 1 and there are 4 distinct blocks, so answer 4.
        
        distinct = set(nums)
        steps = sum(1 for x in distinct if x > k)
        return steps

# -------------------
# Example Test Cases
# -------------------
if __name__ == "__main__":
    sol = Solution()
    # Example 1
    print(sol.minOperations([5,2,5,4,5], 2))  # Expected output: 2
    # Example 2
    print(sol.minOperations([2,1,2], 2))        # Expected output: -1
    # Example 3
    print(sol.minOperations([9,7,5,3], 1))        # Expected output: 4