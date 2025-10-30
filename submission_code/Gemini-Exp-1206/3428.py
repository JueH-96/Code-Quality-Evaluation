class Solution:
    def duplicateNumbersXOR(self, nums: List[int]) -> int:
        counts = {}
        for num in nums:
            counts[num] = counts.get(num, 0) + 1
        
        xor_sum = 0
        for num, count in counts.items():
            if count == 2:
                xor_sum ^= num
        
        return xor_sum