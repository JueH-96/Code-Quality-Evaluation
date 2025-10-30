def solve():
    N = int(input())
    A = list(map(int, input().split()))
    
    MOD = 998244353
    
    from functools import lru_cache
    
    @lru_cache(maxsize=None)
    def count_arithmetic(pos, last_idx, diff, length):
        if length == 0:
            return 1
        if pos >= N:
            return 0
        
        # Option 1: Skip current position
        result = count_arithmetic(pos + 1, last_idx, diff, length)
        
        # Option 2: Include current position
        if last_idx == -1:
            # First element
            result = (result + count_arithmetic(pos + 1, pos, None, length - 1)) % MOD
        elif diff is None:
            # Second element
            result = (result + count_arithmetic(pos + 1, pos, A[pos] - A[last_idx], length - 1)) % MOD
        else:
            # Check if it maintains arithmetic sequence
            if A[pos] - A[last_idx] == diff:
                result = (result + count_arithmetic(pos + 1, pos, diff, length - 1)) % MOD
        
        return result
    
    results = []
    for k in range(1, N + 1):
        count = count_arithmetic(0, -1, None, k)
        results.append(count)
    
    print(' '.join(map(str, results)))

solve()