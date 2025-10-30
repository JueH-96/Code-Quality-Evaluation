from typing import List

class Solution:
    def maxScore(self, a: List[int], b: List[int]) -> int:
        # Initialize variables to store the best scores for different combinations of indices
        best_0 = best_01 = best_012 = float('-inf')
        
        for num in b:
            # Update the best score for the combination of 4 indices
            best_0123 = best_012 * a[3] + num
            
            # Update the best score for the combination of 3 indices
            best_012 = max(best_012, best_01 * a[2] + num)
            
            # Update the best score for the combination of 2 indices
            best_01 = max(best_01, best_0 * a[1] + num)
            
            # Update the best score for the combination of 1 index
            best_0 = max(best_0, a[0] * num)
        
        return best_0123