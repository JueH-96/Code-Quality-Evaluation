class Solution:
    def duplicateNumbersXOR(self, nums: List[int]) -> int:
        # Count frequency of each number
        count = {}
        for num in nums:
            count[num] = count.get(num, 0) + 1
        
        # XOR all numbers that appear exactly twice
        result = 0
        for num, freq in count.items():
            if freq == 2:
                result ^= num
        
        return result