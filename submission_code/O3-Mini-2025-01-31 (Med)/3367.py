from typing import List

class Solution:
    def sumOfEncryptedInt(self, nums: List[int]) -> int:
        total = 0
        for num in nums:
            num_str = str(num)
            largest_digit = max(num_str)
            encrypted_num = int(largest_digit * len(num_str))
            total += encrypted_num
        return total