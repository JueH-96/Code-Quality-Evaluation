from typing import List

class Solution:
    def sumOfEncryptedInt(self, nums: List[int]) -> int:
        total = 0
        for number in nums:
            num_str = str(number)
            max_digit = max(num_str)
            length = len(num_str)
            encrypted_str = max_digit * length
            encrypted_num = int(encrypted_str)
            total += encrypted_num
        return total