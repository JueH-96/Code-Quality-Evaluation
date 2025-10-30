class Solution:
    def duplicateNumbersXOR(self, nums: List[int]) -> int:
        # Create a dictionary to count the frequency of each number
        frequency = {}
        for num in nums:
            frequency[num] = frequency.get(num, 0) + 1
        
        # XOR all numbers that appear twice
        result = 0
        for num, count in frequency.items():
            if count == 2:
                result ^= num
        
        return result