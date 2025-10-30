class Solution:
    def canAliceWin(self, nums: List[int]) -> bool:
        sum_single = sum(x for x in nums if x < 10)
        sum_double = sum(x for x in nums if x >= 10)
        
        # Case 1: Alice chooses single-digit numbers
        if sum_single > sum_double:
            return True
        
        # Case 2: Alice chooses double-digit numbers
        if sum_double > sum_single:
            return True
        
        # If neither choice leads to a strictly greater sum for Alice
        return False