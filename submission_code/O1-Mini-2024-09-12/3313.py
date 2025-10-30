from typing import List

class Solution:
    def maximumStrength(self, nums: List[int], k: int) -> int:
        n = len(nums)
        prefix = [0] * (n + 1)
        for i in range(n):
            prefix[i+1] = prefix[i] + nums[i]
        
        # Compute coefficients
        coeff = []
        for j in range(1, k+1):
            c_j = (k - j + 1) * ((-1) ** (j + 1))
            coeff.append(c_j)
        
        # Initialize DP
        prev = [float('-inf')] * (n + 1)
        prev[0] = 0  # Base case: 0 subarrays selected, strength is 0
        
        for j in range(1, k+1):
            current = [float('-inf')] * (n + 1)
            c_j = coeff[j-1]
            best = float('-inf')
            for i in range(1, n+1):
                # Update best[j] with DP[j-1][i-1] - c_j * prefix[i-1]
                if prev[i-1] - c_j * prefix[i-1] > best:
                    best = prev[i-1] - c_j * prefix[i-1]
                # Update current[j][i]
                option1 = current[i-1]
                option2 = c_j * prefix[i] + best
                current[i] = max(option1, option2)
            prev = current
        
        return int(prev[n])