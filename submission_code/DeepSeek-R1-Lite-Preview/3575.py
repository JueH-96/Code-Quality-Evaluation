class Solution:
    def maxValue(self, nums: List[int], k: int) -> int:
        n = len(nums)
        # Initialize DP1: DP1[m][countA] is a bitmask of achievable A values
        DP1 = [[0] * (k + 1) for _ in range(n + 1)]
        DP1[0][0] = 1  # Only A=0 is achievable with 0 elements
        
        for m in range(1, n + 1):
            for countA in range(0, k + 1):
                if countA <= m:
                    DP1[m][countA] = DP1[m-1][countA]
                    if countA < k:
                        temp = 0
                        for A in range(0, 128):
                            if DP1[m-1][countA] & (1 << A):
                                A_new = A | nums[m-1]
                                temp |= (1 << A_new)
                        DP1[m][countA + 1] |= temp
        
        # Initialize DP2: DP2[m][countB] is a bitmask of achievable B values
        DP2 = [[0] * (k + 1) for _ in range(n + 1)]
        DP2[0][0] = 1  # Only B=0 is achievable with 0 elements
        
        for m in range(1, n + 1):
            for countB in range(0, k + 1):
                if countB <= m:
                    DP2[m][countB] = DP2[m-1][countB]
                    if countB < k:
                        temp = 0
                        for B in range(0, 128):
                            if DP2[m-1][countB] & (1 << B):
                                B_new = B | nums[n - m]
                                temp |= (1 << B_new)
                        DP2[m][countB + 1] |= temp
        
        max_xor = 0
        for s in range(k, n - k + 1):
            # Get possible A values from DP1[s][k]
            A_mask = DP1[s][k]
            # Get possible B values from DP2[n - s][k]
            B_mask = DP2[n - s][k]
            # Iterate through all possible A and B
            for A in range(0, 128):
                if A_mask & (1 << A):
                    for B in range(0, 128):
                        if B_mask & (1 << B):
                            xor_val = A ^ B
                            if xor_val > max_xor:
                                max_xor = xor_val
        return max_xor