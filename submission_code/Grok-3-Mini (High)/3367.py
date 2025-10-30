class Solution:
    def sumOfEncryptedInt(self, nums: List[int]) -> int:
        total_sum = 0
        for num in nums:
            str_num = str(num)
            max_digit = max(int(digit) for digit in str_num)
            num_digits = len(str_num)
            encrypted_num = int(str(max_digit) * num_digits)
            total_sum += encrypted_num
        return total_sum