class Solution:
    def sumOfEncryptedInt(self, nums: List[int]) -> int:
        encrypted_sum = 0
        for num in nums:
            s_num = str(num)
            if not s_num:
                continue
            largest_digit = '0'
            for digit in s_num:
                if digit > largest_digit:
                    largest_digit = digit

            encrypted_num_str = ""
            for _ in range(len(s_num)):
                encrypted_num_str += largest_digit

            if encrypted_num_str:
                encrypted_num = int(encrypted_num_str)
                encrypted_sum += encrypted_num
        return encrypted_sum