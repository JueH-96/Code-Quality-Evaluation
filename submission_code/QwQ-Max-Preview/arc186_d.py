MOD = 998244353

def main():
    import sys
    sys.setrecursionlimit(1 << 25)
    N, *rest = map(int, sys.stdin.read().split())
    A = rest[:N]
    
    if N == 1:
        print(1 if A[0] == 0 else 0)
        return
    
    # Precompute Catalan numbers up to N-1
    max_catalan = N-1
    catalan = [0] * (max_catalan + 1)
    catalan[0] = 1
    for i in range(1, max_catalan + 1):
        catalan[i] = (catalan[i-1] * 2 * (2*i -1) * pow(i+1, MOD-2, MOD)) % MOD
    
    from functools import lru_cache

    @lru_cache(maxsize=None)
    def dfs(pos, eq_flag):
        if pos == N:
            return 1
        
        res = 0
        max_k = min(A[pos] if eq_flag else (N - pos -1), N - pos -1)
        for k in range(0, max_k + 1):
            if k < 0:
                continue
            remaining = N - pos -1
            if remaining < k:
                continue
            
            if eq_flag:
                if k > A[pos]:
                    continue
                new_eq = (k == A[pos])
            else:
                new_eq = False
            
            if k == 0:
                if remaining != 0:
                    continue
                if pos == N-1:
                    res = (res + 1) % MOD
                continue
            
            if remaining == 0:
                continue
            
            if k > remaining:
                continue
            
            if k == 0:
                ways = 1 if remaining ==0 else 0
            else:
                ways = catalan[remaining]  # sum of g(remaining, k) for all possible splits
            
            if not eq_flag or k < A[pos]:
                res = (res + catalan[remaining]) % MOD
            else:
                child_res = split_child(pos +1, k, remaining)
                res = (res + child_res) % MOD
        
        return res
    
    @lru_cache(maxsize=None)
    def split_child(start, k, total_length):
        if k == 0:
            return 0 if total_length !=0 else 1
        
        if start >= N:
            return 0
        
        res = 0
        if k == 1:
            return dfs(start, True)
        
        for s in range(1, total_length - (k-1) +1):
            if start + s > N:
                break
            ways = dfs(start, True)
            next_start = start + s
            remaining = total_length - s
            child_ways = split_child(next_start, k-1, remaining)
            res = (res + ways * child_ways) % MOD
        
        return res
    
    result = dfs(0, True)
    print(result % MOD)

if __name__ == '__main__':
    main()