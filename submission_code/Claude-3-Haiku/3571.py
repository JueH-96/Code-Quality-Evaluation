class Solution:
    def maxPathLength(self, coordinates: List[List[int]], k: int) -> int:
        n = len(coordinates)
        
        # Sort the coordinates by x-coordinate
        coordinates.sort()
        
        # Create a dictionary to store the maximum path length for each coordinate
        dp = {}
        
        def dfs(i):
            if i in dp:
                return dp[i]
            
            max_length = 1
            for j in range(i+1, n):
                if coordinates[j][0] > coordinates[i][0] and coordinates[j][1] > coordinates[i][1]:
                    max_length = max(max_length, 1 + dfs(j))
            
            dp[i] = max_length
            return max_length
        
        max_length = 0
        for i in range(n):
            if i == k:
                max_length = max(max_length, dfs(i))
        
        return max_length