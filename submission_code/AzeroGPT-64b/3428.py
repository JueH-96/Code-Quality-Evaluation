class Solution:
    def duplicateNumbersXOR(self, nums: List[int]) -> int:
        xor = 0
        freq = {}
        for num in nums:
            if num not in freq:
                freq[num] = 1
            else:
                freq[num] += 1
                
            if freq[num] % 2 != 0:
                xor ^= num
            elif freq[num] % 2 == 0 and freq[num] != 1:
                xor ^= num
        
        return xor