class Solution:
    def numberOfPermutations(self, n: int, requirements: List[List[int]]) -> int:
        MOD = 10 ** 9 + 7
        
        # Sort the requirements by end index
        requirements.sort(key=lambda x: x[0])
        
        # Create a dp table to store the number of permutations that satisfy the requirements up to each index
        dp = [[0] * (max(req[1] for req in requirements) + 1) for _ in range(n + 1)]
        
        # Initialize the base case
        dp[0][0] = 1
        
        for i in range(1, n + 1):
            for j in range(max(req[1] for req in requirements) + 1):
                # Include the current element
                dp[i][j] = (dp[i][j] + dp[i - 1][j]) % MOD
                
                # Check if there are any requirements that end at the current index
                for end, cnt in requirements:
                    if end == i - 1:
                        # Update the dp table based on the inversion count
                        dp[i][(j + cnt) % (max(req[1] for req in requirements) + 1)] = (dp[i][(j + cnt) % (max(req[1] for req in requirements) + 1)] + dp[i - 1][j]) % MOD
        
        # Return the number of permutations that satisfy all requirements
        return dp[n][0]