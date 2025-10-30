class Solution:
    def minMaxSums(self, nums: List[int], k: int) -> int:
        MOD = 10**9 + 7
        n = len(nums)
        nums.sort()
        
        # Precompute factorials and inverse factorials for combinations
        fact = [1] * (n + 1)
        for i in range(1, n + 1):
            fact[i] = fact[i-1] * i % MOD
        
        # Precompute powers of 2
        pow2 = [1] * n
        for i in range(1, n):
            pow2[i] = pow2[i-1] * 2 % MOD
        
        # Precompute binomial coefficients C(n, r) for all needed values
        # We'll use Pascal's triangle approach
        C = [[0] * (k + 1) for _ in range(n + 1)]
        for i in range(n + 1):
            C[i][0] = 1
            for j in range(1, min(i + 1, k + 1)):
                C[i][j] = (C[i-1][j-1] + C[i-1][j]) % MOD
        
        result = 0
        
        # For each possible minimum element
        for i in range(n):
            # For each possible maximum element >= minimum
            for j in range(i, n):
                min_val = nums[i]
                max_val = nums[j]
                contribution = (min_val + max_val) % MOD
                
                # Count valid subsequences
                count = 0
                
                if i == j:
                    # Only one element, it's both min and max
                    count = 1
                else:
                    # We must include nums[i] and nums[j]
                    # We can choose from the middle elements (j - i - 1 elements)
                    middle = j - i - 1
                    
                    # We need at least 2 elements (i and j) and at most k elements
                    for size in range(2, min(k + 1, j - i + 2)):
                        # We already have 2 elements (i and j)
                        # We need to choose (size - 2) from middle elements
                        if size - 2 <= middle:
                            count = (count + C[middle][size - 2]) % MOD
                
                result = (result + contribution * count) % MOD
        
        return result