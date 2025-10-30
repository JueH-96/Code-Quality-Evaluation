class Solution:
    def maxPathLength(self, coordinates: List[List[int]], k: int) -> int:
        n = len(coordinates)
        dp_forward = [1] * n
        dp_backward = [1] * n

        for i in range(n):
            for j in range(n):
                if i != j:
                    if coordinates[j][0] < coordinates[i][0] and coordinates[j][1] < coordinates[i][1]:
                        dp_forward[i] = max(dp_forward[i], dp_forward[j] + 1)

        for i in range(n):
            for j in range(n):
                if i != j:
                    if coordinates[i][0] < coordinates[j][0] and coordinates[i][1] < coordinates[j][1]:
                        dp_backward[i] = max(dp_backward[i], dp_backward[j] + 1)
                        
        return dp_forward[k] + dp_backward[k] - 1