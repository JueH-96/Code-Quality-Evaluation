class Solution:
    def sumOfEncryptedInt(self, nums: List[int]) -> int:
        def encrypt(x):
            s = str(x)
            max_digit = max(s)
            return int(max_digit * len(s))
        
        total = 0
        for num in nums:
            total += encrypt(num)
        return total