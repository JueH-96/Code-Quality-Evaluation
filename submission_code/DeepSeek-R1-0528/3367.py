class Solution:
    def sumOfEncryptedInt(self, nums: List[int]) -> int:
        total = 0
        for num in nums:
            s = str(num)
            m = max(s)
            encrypted_num = int(m * len(s))
            total += encrypted_num
        return total