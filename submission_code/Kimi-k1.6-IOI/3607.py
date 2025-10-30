class Solution:
    def minOperations(self, nums: List[int]) -> int:
        max_num = 10**6
        # Precompute smallest prime factors using sieve of Eratosthenes
        spf = list(range(max_num + 2))  # Initialize spf[i] = i for all i
        spf[0] = spf[1] = 0
        for i in range(2, int(max_num**0.5) + 1):
            if spf[i] == i:  # i is a prime
                for j in range(i * i, max_num + 1, i):
                    if spf[j] == j:
                        spf[j] = i
        
        n = len(nums)
        if n == 0:
            return 0
        
        next_val = float('inf')
        total_ops = 0
        
        for i in reversed(range(n)):
            x = nums[i]
            if x == 1:
                current_values = [1]
                current_ops = [0]
            elif spf[x] == x:
                current_values = [x]
                current_ops = [0]
            else:
                p = spf[x]
                current_values = [x, p]
                current_ops = [0, 1]
            
            found = False
            for val, op in zip(current_values, current_ops):
                if val <= next_val:
                    total_ops += op
                    next_val = val
                    found = True
                    break
            if not found:
                return -1
        
        return total_ops