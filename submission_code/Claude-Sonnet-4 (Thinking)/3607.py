class Solution:
    def minOperations(self, nums: List[int]) -> int:
        if len(nums) == 1:
            return 0
        
        # Precompute smallest prime factors using sieve
        max_val = max(nums)
        spf = list(range(max_val + 1))  # spf[i] = smallest prime factor of i
        for i in range(2, int(max_val**0.5) + 1):
            if spf[i] == i:  # i is prime
                for j in range(i * i, max_val + 1, i):
                    if spf[j] == j:
                        spf[j] = i
        
        def get_reachable_values(n):
            values = [(n, 0)]  # (value, operations)
            current = n
            operations = 0
            
            while current > 1:
                if spf[current] == current:  # current is prime
                    break  # Can't reduce further (would divide by 1)
                gpd = current // spf[current]  # greatest proper divisor
                next_val = current // gpd       # result of the operation
                current = next_val
                operations += 1
                values.append((current, operations))
            
            return values
        
        n = len(nums)
        
        # Get reachable values for each number
        reachable = []
        for num in nums:
            reachable.append(get_reachable_values(num))
        
        # DP: dp[(i,j)] = min operations to make first i+1 elements non-decreasing
        # where nums[i] is set to reachable[i][j][0]
        dp = {}
        
        # Base case: first element
        for j, (val, ops) in enumerate(reachable[0]):
            dp[(0, j)] = ops
        
        # Fill DP table
        for i in range(1, n):
            for j, (val, ops) in enumerate(reachable[i]):
                dp[(i, j)] = float('inf')
                for k, (prev_val, prev_ops) in enumerate(reachable[i-1]):
                    if prev_val <= val:
                        dp[(i, j)] = min(dp[(i, j)], dp[(i-1, k)] + ops)
        
        # Find the answer
        result = float('inf')
        for j in range(len(reachable[n-1])):
            if (n-1, j) in dp:
                result = min(result, dp[(n-1, j)])
        
        return result if result != float('inf') else -1