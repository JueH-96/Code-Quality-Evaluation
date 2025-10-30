class Solution:
    def sumOfEncryptedInt(self, nums: List[int]) -> int:
        total_sum = 0
        for num in nums:
            max_digit = max(str(num))
            encrypted_num = int(max_digit * len(str(num)))
            total_sum += encrypted_num
        return total_sum