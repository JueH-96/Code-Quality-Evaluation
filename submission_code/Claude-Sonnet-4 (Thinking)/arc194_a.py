import sys
sys.setrecursionlimit(200000)

def main():
    N = int(input())
    A = list(map(int, input().split()))
    
    from functools import lru_cache
    
    @lru_cache(None)
    def dp(i, stack):
        if i == N:
            return sum(stack)
        
        # Option 1: Append A[i]
        result1 = dp(i + 1, stack + (A[i],))
        
        # Option 2: Delete last element (if possible)
        result2 = float('-inf')
        if stack:
            result2 = dp(i + 1, stack[:-1])
        
        return max(result1, result2)
    
    return dp(0, ())

print(main())