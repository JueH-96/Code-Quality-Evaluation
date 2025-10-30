from typing import List

class Solution:
    def maximumEnergy(self, energy: List[int], k: int) -> int:
        n = len(energy)
        # dp[i] holds the maximum energy that can be gained starting from magician i
        dp = [0] * n
        
        # Process the magicians in reverse order
        for i in range(n - 1, -1, -1):
            if i + k < n:
                dp[i] = energy[i] + dp[i + k]
            else:
                dp[i] = energy[i]
        
        # The answer is the maximum energy among all potential starting positions
        return max(dp)

# Example usage:
if __name__ == "__main__":
    sol = Solution()
    print(sol.maximumEnergy([5, 2, -10, -5, 1], 3))  # Output: 3
    print(sol.maximumEnergy([-2, -3, -1], 2))          # Output: -1