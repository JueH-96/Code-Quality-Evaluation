from typing import List
import math

class Solution:
    def maxScore(self, a: List[int], b: List[int]) -> int:
        # We are to choose indices i_0, i_1, i_2, i_3 (with i0 < i1 < i2 < i3)
        # so that the score = a[0]*b[i_0] + a[1]*b[i_1] + a[2]*b[i_2] + a[3]*b[i_3] is maximum.
        #
        # We can solve this using dynamic programming. Define dp[k] as the maximum score
        # we can obtain by choosing k elements (in order) from b.
        # We start with dp[0] = 0 and dp[1..4] = -infinity.
        # We then iterate through b in order. For each element in b:
        #    for k from 3 downto 0:
        #         dp[k+1] = max(dp[k+1], dp[k] + a[k] * current_b_value)
        # Finally, dp[4] holds our answer.
        
        # Number of selections required is 4.
        dp = [-math.inf] * 5
        dp[0] = 0
        
        for value in b:
            # Update in reverse order to ensure each element in b is used only once in the sequence.
            for k in range(3, -1, -1):
                if dp[k] != -math.inf:
                    dp[k + 1] = max(dp[k + 1], dp[k] + a[k] * value)
                    
        return dp[4]
        
# Example usage:
if __name__ == "__main__":
    solution = Solution()
    
    # Example 1:
    a1 = [3, 2, 5, 6]
    b1 = [2, -6, 4, -5, -3, 2, -7]
    print(solution.maxScore(a1, b1))  # Expected output: 26
    
    # Example 2:
    a2 = [-1, 4, 5, -2]
    b2 = [-5, -1, -3, -2, -4]
    print(solution.maxScore(a2, b2))  # Expected output: -1