class Solution:
    def sumOfEncryptedInt(self, nums: List[int]) -> int:
        encrypted_nums = []
        for num in nums:
            s_num = str(num)
            max_digit = '0'
            for digit in s_num:
                if digit > max_digit:
                    max_digit = digit
            encrypted_num_str = max_digit * len(s_num)
            encrypted_nums.append(int(encrypted_num_str))
        return sum(encrypted_nums)