from typing import List

class Solution:
    def findKOr(self, nums: List[int], k: int) -> int:
        result = 0
        for i in range(32):  # Iterate over each bit position (0 to 31)
            count = sum((num >> i) & 1 for num in nums)  # Count how many numbers have the i-th bit set
            if count >= k:  # If at least k numbers have the i-th bit set
                result |= (1 << i)  # Set the i-th bit in the result
        return result