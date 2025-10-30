class Solution:
    def sumOfEncryptedInt(self, nums: List[int]) -> int:
        sum_encrypted = 0
        for num in nums:
            s_num = str(num)
            max_digit = max(s_num)
            encrypted_num = int(max_digit * len(s_num))
            sum_encrypted += encrypted_num
        return sum_encrypted