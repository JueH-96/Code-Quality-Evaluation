from collections import defaultdict

MOD = 10**9 + 7

class Solution:
    def numberOfPermutations(self, n: int, requirements: List[List[int]]) -> int:
        req = {}
        for end, cnt in requirements:
            req[end] = cnt
        
        # Check if n-1 is in req as per problem statement
        if (n - 1) not in req:
            return 0
        
        # Initialize DP: dp_prev represents the state before inserting i
        dp_prev = defaultdict(int)
        dp_prev[0] = 1  # Starting with an empty permutation
        
        for i in range(1, n + 1):
            dp_curr = defaultdict(int)
            current_max = i - 1  # The number being inserted is (i-1)
            
            for k in dp_prev:
                # Inserting into positions 0 to (i-1)
                for j in range(i):
                    new_k = k + (current_max - j)
                    dp_curr[new_k] = (dp_curr[new_k] + dp_prev[k]) % MOD
            
            # Check if the current end (i-1) has a requirement
            if (i - 1) in req:
                required = req[i - 1]
                # Only keep the required inversion count
                count = dp_curr.get(required, 0)
                dp_curr = defaultdict(int)
                if count > 0:
                    dp_curr[required] = count % MOD
            
            dp_prev = dp_curr
        
        return dp_prev.get(req[n - 1], 0) % MOD