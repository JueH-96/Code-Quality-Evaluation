class Solution:
    def sumOfEncryptedInt(self, nums: List[int]) -> int:
        total_sum = 0
        for num in nums:
            str_num = str(num)
            max_digit = max(str_num)
            encrypted_num = int(max_digit * len(str_num))
            total_sum += encrypted_num
        return total_sum