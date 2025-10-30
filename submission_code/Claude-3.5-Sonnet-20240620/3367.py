class Solution:
    def sumOfEncryptedInt(self, nums: List[int]) -> int:
        def encrypt(num):
            max_digit = max(int(d) for d in str(num))
            return int(str(max_digit) * len(str(num)))
        
        return sum(encrypt(num) for num in nums)