class Solution:
    def duplicateNumbersXOR(self, nums: List[int]) -> int:
        from collections import Counter
        
        # Count the frequency of each number in the array
        freq = Counter(nums)
        
        # Initialize the result for XOR operation
        result = 0
        
        # Iterate through the frequency dictionary
        for num, count in freq.items():
            if count == 2:
                result ^= num
        
        return result