class Solution:
    def sumOfEncryptedInt(self, nums: List[int]) -> int:
        def encrypt(x):
            s_x = str(x)
            max_digit = '0'
            for digit in s_x:
                if digit > max_digit:
                    max_digit = digit
            encrypted_s = ""
            for _ in range(len(s_x)):
                encrypted_s += max_digit
            return int(encrypted_s)

        encrypted_nums_sum = 0
        for num in nums:
            encrypted_nums_sum += encrypt(num)
        return encrypted_nums_sum