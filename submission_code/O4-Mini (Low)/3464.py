from typing import List

class Solution:
    def maximumTotalCost(self, nums: List[int]) -> int:
        n = len(nums)
        # Compute the two alternating prefix sums:
        # A[i] = nums[i] * (-1)^i, B[i] = nums[i] * (-1)^(i+1)
        prefA = [0] * n
        prefB = [0] * n
        a = 1
        b = -1
        # a = (-1)^i, b = (-1)^(i+1) for i=0
        for i in range(n):
            prefA[i] = nums[i] * a + (prefA[i-1] if i > 0 else 0)
            prefB[i] = nums[i] * b + (prefB[i-1] if i > 0 else 0)
            # flip signs for next index
            a = -a
            b = -b
        
        # dp_next will hold dp[i+1], initially dp[n] = 0
        dp_next = 0
        # bestA_next = max_{j>i}( prefA[j] + dp[j+1] )
        # bestB_next = max_{j>i}( prefB[j] + dp[j+1] )
        # Initialize with -infinity
        NEG_INF = float('-inf')
        bestA_next = NEG_INF
        bestB_next = NEG_INF
        
        # We'll fill dp backwards; dp[i] = max total cost from i..end
        dp_i = 0  # placeholder
        for i in range(n-1, -1, -1):
            # First incorporate j = i into bestA and bestB
            # so bestX_i = max(bestX_{i+1}, prefX[i] + dp[i+1])
            curA = prefA[i] + dp_next
            curB = prefB[i] + dp_next
            bestA_i = max(bestA_next, curA)
            bestB_i = max(bestB_next, curB)
            
            # Compute dp[i]
            if i % 2 == 0:
                # starting index even => use A
                C0 = prefA[i-1] if i > 0 else 0
                dp_i = bestA_i - C0
            else:
                # starting index odd => use B
                C0 = prefB[i-1] if i > 0 else 0
                dp_i = bestB_i - C0
            
            # shift for next iteration
            dp_next = dp_i
            bestA_next = bestA_i
            bestB_next = bestB_i
        
        return dp_i