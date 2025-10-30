from typing import List

class Solution:
    def constructTransformedArray(self, nums: List[int]) -> List[int]:
        n = len(nums)
        result = [0] * n  # Initialize the result array with zeros, same size as nums

        # Iterate through each element of the input array
        for i in range(n):
            if nums[i] == 0:
                # Rule: If nums[i] is 0, set result[i] to 0
                result[i] = 0
            else:
                # Calculate the target index after moving steps in a circular array.
                # The expression (i + nums[i]) naturally handles both positive (move right)
                # and negative (move left) steps.
                # Python's % operator ensures the result is always non-negative
                # and within the bounds [0, n-1] for positive n, which is perfect
                # for circular array indexing.
                landed_index = (i + nums[i]) % n
                
                # As per the examples, "value of the index where you land"
                # refers to the value in the original nums array at that landed index.
                result[i] = nums[landed_index]
        
        return result