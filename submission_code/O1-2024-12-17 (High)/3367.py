class Solution:
    def sumOfEncryptedInt(self, nums: List[int]) -> int:
        total_sum = 0
        for x in nums:
            x_str = str(x)
            max_digit = max(x_str)
            encrypted_str = max_digit * len(x_str)
            total_sum += int(encrypted_str)
        return total_sum