class Solution:
    def duplicateNumbersXOR(self, nums: List[int]) -> int:
        from collections import Counter
        
        # Count the occurrences of each number
        count = Counter(nums)
        
        # Initialize result for XOR operation
        result = 0
        
        # XOR all numbers that appear twice
        for num, freq in count.items():
            if freq == 2:
                result ^= num
        
        return result