class Solution:
    def maxGoodNumber(self, nums: List[int]) -> int:
        # Convert each number to binary and remove the '0b' prefix
        bin_nums = [bin(num)[2:] for num in nums]
        # Sort the binary strings in descending order
        bin_nums.sort(reverse=True)
        # Concatenate the binary strings
        result = ''.join(bin_nums)
        # Convert the result back to an integer and return
        return int(result, 2)