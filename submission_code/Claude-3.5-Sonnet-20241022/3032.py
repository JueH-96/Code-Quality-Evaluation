class Solution:
    def getMaxFunctionValue(self, receiver: List[int], k: int) -> int:
        n = len(receiver)
        # Calculate binary length of k
        binary_k = bin(k)[2:]
        m = len(binary_k)
        
        # dp[j][i] stores the 2^j-th receiver from player i
        dp = [[0] * n for _ in range(m)]
        # sum[j][i] stores sum of all ids in path of length 2^j from player i
        sums = [[0] * n for _ in range(m)]
        
        # Initialize for 2^0 = 1 step
        for i in range(n):
            dp[0][i] = receiver[i]
            sums[0][i] = receiver[i]
            
        # Build dp arrays for powers of 2
        for j in range(1, m):
            for i in range(n):
                # Combine two paths of length 2^(j-1)
                prev = dp[j-1][i]
                dp[j][i] = dp[j-1][prev]
                sums[j][i] = sums[j-1][i] + sums[j-1][prev]
        
        result = 0
        # Try each starting position
        for start in range(n):
            curr_pos = start
            total = start
            
            # Process each bit of k from left to right
            for j in range(m):
                if binary_k[j] == '1':
                    total += sums[m-1-j][curr_pos]
                    curr_pos = dp[m-1-j][curr_pos]
            
            result = max(result, total)
            
        return result