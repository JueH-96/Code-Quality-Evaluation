class Solution:
    def sumOfEncryptedInt(self, nums: List[int]) -> int:
        sum_encrypted = 0
        for num in nums:
            s_num = str(num)
            if not s_num:
                continue
            max_digit = 0
            for digit in s_num:
                max_digit = max(max_digit, int(digit))
            encrypted_num = int(str(max_digit) * len(s_num))
            sum_encrypted += encrypted_num
        return sum_encrypted