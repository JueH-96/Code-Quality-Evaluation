class Solution:
    def maxGoodNumber(self, nums: List[int]) -> int:
        # Convert each number to its binary representation
        binary_nums = [format(num, 'b') for num in nums]
        
        # Sort the binary representations in descending order
        binary_nums.sort(key=lambda x: len(x), reverse=True)
        
        # Concatenate the binary representations
        result = ''.join(binary_nums)
        
        # Convert the binary representation back to decimal and return the result
        return int(result, 2)