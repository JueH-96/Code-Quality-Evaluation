class Solution:
    def maxOperations(self, s: str) -> int:
        # Collect the indices of '1's in the string
        ones = [i for i, ch in enumerate(s) if ch == '1']
        k = len(ones)
        # If there are no '1's, no operations are possible
        if k == 0:
            return 0
        
        n = len(s)
        ans = 0
        # We number the zero‐blocks z_1, z_2, ..., z_k where:
        #  z_j for j=1..k-1 is the zeros between ones[j-1] and ones[j],
        #  z_k is the trailing zeros after the last '1'.
        # One may perform an operation on block j if z_j > 0,
        # and each such block contributes j to the max count.
        for j in range(1, k + 1):
            if j < k:
                # zeros between ones[j-1] and ones[j]
                zero_count = ones[j] - ones[j-1] - 1
            else:
                # trailing zeros after the last '1'
                zero_count = n - 1 - ones[j-1]
            
            if zero_count > 0:
                ans += j
        
        return ans