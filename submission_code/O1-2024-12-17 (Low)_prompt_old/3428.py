class Solution:
    def duplicateNumbersXOR(self, nums: List[int]) -> int:
        freq = {}
        for num in nums:
            freq[num] = freq.get(num, 0) + 1
        
        result = 0
        for num, count in freq.items():
            if count == 2:
                result ^= num
        
        return result