class Solution:
    def canTraverseAllPairs(self, nums: List[int]) -> bool:
        def find(x):
            if parent[x] != x:
                parent[x] = find(parent[x])
            return parent[x]
        
        def union(x, y):
            px, py = find(x), find(y)
            if px != py:
                parent[px] = py
        
        n = len(nums)
        if n == 1:
            return True
        
        parent = list(range(n))
        factor_index = {}
        
        for i, num in enumerate(nums):
            if num == 1:
                return False
            
            f = 2
            while f * f <= num:
                if num % f == 0:
                    if f in factor_index:
                        union(i, factor_index[f])
                    else:
                        factor_index[f] = i
                    while num % f == 0:
                        num //= f
                f += 1
            
            if num > 1:
                if num in factor_index:
                    union(i, factor_index[num])
                else:
                    factor_index[num] = i
        
        return len(set(find(i) for i in range(n))) == 1