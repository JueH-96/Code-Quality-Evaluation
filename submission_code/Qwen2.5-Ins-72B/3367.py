class Solution:
    def sumOfEncryptedInt(self, nums: List[int]) -> int:
        def encrypt(x: int) -> int:
            max_digit = max(int(d) for d in str(x))
            return int(str(max_digit) * len(str(x)))
        
        return sum(encrypt(num) for num in nums)