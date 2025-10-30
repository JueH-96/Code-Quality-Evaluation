class Solution:
    def maxGoodNumber(self, nums: List[int]) -> int:
        # Generate all permutations of the input list
        permutations = list(itertools.permutations(nums))
        
        # Initialize the maximum number to 0
        max_num = 0
        
        # Iterate over all permutations
        for perm in permutations:
            # Convert each number in the permutation to its binary representation
            # and concatenate them into a single string
            binary_str = ''.join(format(num, 'b') for num in perm)
            
            # Convert the binary string back to an integer
            num = int(binary_str, 2)
            
            # Update the maximum number if the current number is greater
            max_num = max(max_num, num)
        
        return max_num