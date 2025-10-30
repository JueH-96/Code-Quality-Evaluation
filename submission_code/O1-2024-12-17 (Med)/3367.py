class Solution:
    def sumOfEncryptedInt(self, nums: List[int]) -> int:
        def encrypt(x: int) -> int:
            s = str(x)
            max_digit = max(s)
            return int(max_digit * len(s))
        
        return sum(encrypt(num) for num in nums)