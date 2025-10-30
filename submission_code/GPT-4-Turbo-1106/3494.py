class Solution:
    def minimumCost(self, m: int, n: int, horizontalCut: List[int], verticalCut: List[int]) -> int:
        # Sort the cuts in descending order to minimize the cost
        horizontalCut.sort(reverse=True)
        verticalCut.sort(reverse=True)
        
        # Initialize the cost and the current size of the cake
        cost = 0
        current_m, current_n = m, n
        
        # Perform cuts while there are still cuts to be made
        while horizontalCut or verticalCut:
            # Determine the next cut direction based on the maximum cost
            if horizontalCut and (not verticalCut or horizontalCut[0] >= verticalCut[0]):
                # Perform a horizontal cut
                cost += horizontalCut.pop(0) * current_n
                current_m -= 1
            else:
                # Perform a vertical cut
                cost += verticalCut.pop(0) * current_m
                current_n -= 1
        
        return cost