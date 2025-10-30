def solve():
    N = int(input())
    A = list(map(int, input().split()))
    MOD = 998244353
    
    from functools import lru_cache
    
    @lru_cache(maxsize=None)
    def dp(pos, needed, is_equal):
        # pos: current position in sequence
        # needed: number of Polish subsequences we need to form from here
        # is_equal: whether we must stay <= A[pos:]
        
        if needed == 0:
            return 1 if pos == N else 0
        if pos >= N:
            return 0
        
        result = 0
        max_val = A[pos] if is_equal else N - 1
        
        for val in range(0, min(max_val + 1, N)):
            # Start a new Polish subsequence with first element val
            # After this element, we need val Polish subsequences to complete this one
            # Plus needed-1 more subsequences after this entire subsequence
            
            new_equal = is_equal and (val == A[pos])
            ways = dp(pos + 1, val + needed - 1, new_equal)
            result = (result + ways) % MOD
        
        return result
    
    # We need to form exactly 1 Polish sequence
    result = dp(0, 1, True)
    print(result)

solve()