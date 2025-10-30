class Solution:
    def sumOfEncryptedInt(self, nums: List[int]) -> int:
        total_sum = 0
        for num in nums:
            s_num = str(num)
            max_digit = '0'
            for digit in s_num:
                if digit > max_digit:
                    max_digit = digit
            encrypted_num = int(max_digit * len(s_num))
            total_sum += encrypted_num
        return total_sum