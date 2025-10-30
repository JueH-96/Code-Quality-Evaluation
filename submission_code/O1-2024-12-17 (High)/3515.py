class Solution:
    def canAliceWin(self, nums: List[int]) -> bool:
        sum_of_singles = sum(n for n in nums if n < 10)
        sum_of_doubles = sum(n for n in nums if n >= 10)
        # Alice can choose single-digit numbers or double-digit numbers
        return sum_of_singles > sum_of_doubles or sum_of_doubles > sum_of_singles