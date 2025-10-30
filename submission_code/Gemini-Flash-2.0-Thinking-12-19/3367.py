class Solution:
    def sumOfEncryptedInt(self, nums: List[int]) -> int:
        encrypted_nums = []
        for num in nums:
            s_num = str(num)
            max_digit = '0'
            for digit in s_num:
                if digit > max_digit:
                    max_digit = digit
            encrypted_s_num = ""
            for _ in range(len(s_num)):
                encrypted_s_num += max_digit
            encrypted_nums.append(int(encrypted_s_num))
        return sum(encrypted_nums)