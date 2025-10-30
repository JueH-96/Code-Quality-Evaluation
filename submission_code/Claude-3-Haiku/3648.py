class Solution:
    def maxCollectedFruits(self, fruits: List[List[int]]) -> int:
        n = len(fruits)
        
        # Helper function to get the maximum fruits collected by a single child
        def max_fruits_collected(i, j, di, dj):
            total = 0
            while 0 <= i < n and 0 <= j < n:
                total += fruits[i][j]
                fruits[i][j] = 0
                i += di
                j += dj
            return total
        
        # Collect fruits for each child
        child1 = max_fruits_collected(0, 0, 1, 1)
        child2 = max_fruits_collected(0, n-1, 1, -1)
        child3 = max_fruits_collected(n-1, 0, -1, 1)
        
        return child1 + child2 + child3