class Solution:
    def maxScore(self, n: int, k: int, stayScore: List[List[int]], travelScore: List[List[int]]) -> int:
        # Initialize the DP array for the first day
        previous = [0] * n
        for j in range(n):
            max_travel = max(travelScore[i][j] for i in range(n))
            previous[j] = max(stayScore[0][j], max_travel)
        
        # Iterate through each subsequent day
        for day in range(1, k):
            current = [0] * n
            for j in range(n):
                # Option 1: Stay in the current city
                stay = previous[j] + stayScore[day][j]
                # Option 2: Move to city j from any other city
                move = max(previous[m] + travelScore[m][j] for m in range(n))
                current[j] = max(stay, move)
            previous = current  # Update previous to current for the next iteration
        
        return max(previous)