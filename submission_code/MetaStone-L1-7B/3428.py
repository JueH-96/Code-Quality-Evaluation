class Solution:
    def duplicateNumbersXOR(self, nums: List[int]) -> int:
        freq = {}
        for num in nums:
            if num in freq:
                freq[num] += 1
            else:
                freq[num] = 1
        duplicates = [k for k, v in freq.items() if v == 2]
        xor = 0
        for num in duplicates:
            xor ^= num
        return xor