class Solution:
    def duplicateNumbersXOR(self, nums: List[int]) -> int:
        # Count frequency of each number
        freq = {}
        for num in nums:
            freq[num] = freq.get(num, 0) + 1
        
        # XOR all numbers that appear exactly twice
        result = 0
        for num, count in freq.items():
            if count == 2:
                result ^= num
        
        return result