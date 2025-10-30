class Solution:
    def numberOfPermutations(self, n: int, requirements: List[List[int]]) -> int:
        MOD = 10**9 + 7
        
        # Convert requirements to a dictionary for easier lookup
        req_dict = {end: cnt for end, cnt in requirements}
        
        # Memoization
        memo = {}
        perm = [-1] * n
        
        def inv_count(end):
            count = 0
            for i in range(end):
                for j in range(i + 1, end + 1):
                    if perm[i] > perm[j]:
                        count += 1
            return count
        
        def f(pos, mask):
            if pos == n:
                return 1
            
            if (pos, mask) in memo:
                return memo[(pos, mask)]
            
            # Check requirements for prefixes we've already filled
            for end in range(pos):
                if end in req_dict and inv_count(end) != req_dict[end]:
                    memo[(pos, mask)] = 0
                    return 0
            
            result = 0
            
            for k in range(n):
                if not (mask & (1 << k)):
                    new_mask = mask | (1 << k)
                    perm[pos] = k
                    
                    # Check if placing element k at position pos would satisfy the requirements
                    valid = True
                    if pos in req_dict and inv_count(pos) != req_dict[pos]:
                        valid = False
                    
                    if valid:
                        result = (result + f(pos + 1, new_mask)) % MOD
                    
                    # Backtrack
                    perm[pos] = -1
            
            memo[(pos, mask)] = result
            return result
        
        return f(0, 0)