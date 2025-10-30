import collections
from typing import List

class Solution:
    def duplicateNumbersXOR(self, nums: List[int]) -> int:
        
        # Use collections.Counter to find frequencies of each number in nums.
        # Example: nums = [1,2,1,3] -> counts will be {1: 2, 2: 1, 3: 1}
        # Example: nums = [1,2,2,1] -> counts will be {1: 2, 2: 2}
        counts = collections.Counter(nums)
        
        xor_result = 0
        
        # Iterate through the items (number, frequency_count) in the Counter object.
        for num_value, frequency in counts.items():
            # According to the problem statement, each number appears either once or twice.
            # We are interested in numbers that appear twice.
            if frequency == 2:
                # If the number appeared twice, XOR it with our accumulated result.
                xor_result ^= num_value
                
        # If no number appears twice, no number's frequency will be 2.
        # In this case, the loop will not perform any XOR operations on xor_result,
        # and it will remain its initial value of 0. This matches the requirement:
        # "or 0 if no number appears twice".
        return xor_result