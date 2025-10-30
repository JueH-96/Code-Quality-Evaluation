class Solution:
    def sumOfEncryptedInt(self, nums: List[int]) -> int:
        total = 0
        for num in nums:
            max_digit = max(map(int, str(num)))
            encrypted_num = int(''.join(str(max_digit) for _ in range(len(str(num)))))
            total += encrypted_num
        return total