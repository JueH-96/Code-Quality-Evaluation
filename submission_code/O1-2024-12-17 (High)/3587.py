class Solution:
    def maxScore(self, n: int, k: int, stayScore: List[List[int]], travelScore: List[List[int]]) -> int:
        # Edge case: if k == 0 (though per constraints k>=1), no days => 0 points
        if k == 0:
            return 0
        
        # dp[i][c] = maximum points achievable ending day i in city c
        dp = [[0] * n for _ in range(k)]
        
        # Initialize dp for day 0
        # We can start in any city for free, then choose to either stay or travel on day 0
        for c in range(n):
            # Staying: stayScore[0][c]
            stay_val = stayScore[0][c]
            # Traveling from some other city x != c: travelScore[x][c]
            # but if n=1, there's no x!=c, so just stay
            travel_val = float('-inf')
            if n > 1:
                for x in range(n):
                    if x != c:
                        travel_val = max(travel_val, travelScore[x][c])
            
            dp[0][c] = max(stay_val, travel_val if travel_val != float('-inf') else stay_val)
        
        # Fill dp for days 1..k-1
        for i in range(1, k):
            for c in range(n):
                # Option 1: stay in the same city c
                stay_val = dp[i-1][c] + stayScore[i][c]
                
                # Option 2: move from a different city c' to c
                move_val = float('-inf')
                for c_prev in range(n):
                    if c_prev != c:
                        move_val = max(move_val, dp[i-1][c_prev] + travelScore[c_prev][c])
                
                dp[i][c] = max(stay_val, move_val)
        
        # The answer is the best we can achieve after the last day among all cities
        return max(dp[k-1])