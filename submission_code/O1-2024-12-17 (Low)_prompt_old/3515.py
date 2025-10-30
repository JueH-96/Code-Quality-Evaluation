class Solution:
    def canAliceWin(self, nums: List[int]) -> bool:
        # Calculate the sum of single-digit numbers (1-9)
        sum_singles = sum(num for num in nums if num < 10)
        # Calculate the sum of double-digit numbers (10-99)
        sum_doubles = sum(num for num in nums if num >= 10)

        # Check if choosing all single-digit numbers gives Alice more than the sum of double-digit numbers
        if sum_singles > sum_doubles:
            return True
        # Check if choosing all double-digit numbers gives Alice more than the sum of single-digit numbers
        if sum_doubles > sum_singles:
            return True
        
        # If neither choice leads to a strictly greater sum for Alice, return False
        return False