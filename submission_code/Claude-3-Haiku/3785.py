class Solution:
    def countArrays(self, original: List[int], bounds: List[List[int]]) -> int:
        n = len(original)
        dp = [[0] * 2 for _ in range(n)]
        
        # Initialize the first row of the dp table
        dp[0][0] = 1 if bounds[0][0] <= original[0] <= bounds[0][1] else 0
        dp[0][1] = 1 - dp[0][0]
        
        for i in range(1, n):
            diff = original[i] - original[i-1]
            for j in range(2):
                for k in range(2):
                    if j == 0 and k == 0:
                        if bounds[i][0] <= original[i] <= bounds[i][1] and bounds[i-1][0] <= original[i-1] + diff <= bounds[i-1][1]:
                            dp[i][0] += dp[i-1][0]
                    elif j == 0 and k == 1:
                        if bounds[i][0] <= original[i] <= bounds[i][1] and not (bounds[i-1][0] <= original[i-1] + diff <= bounds[i-1][1]):
                            dp[i][0] += dp[i-1][1]
                    elif j == 1 and k == 0:
                        if not (bounds[i][0] <= original[i] <= bounds[i][1]) and bounds[i-1][0] <= original[i-1] + diff <= bounds[i-1][1]:
                            dp[i][1] += dp[i-1][0]
                    elif j == 1 and k == 1:
                        if not (bounds[i][0] <= original[i] <= bounds[i][1]) and not (bounds[i-1][0] <= original[i-1] + diff <= bounds[i-1][1]):
                            dp[i][1] += dp[i-1][1]
        
        return sum(dp[n-1])