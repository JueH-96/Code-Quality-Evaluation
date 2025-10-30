class Solution:
    def countComponents(self, nums: List[int], threshold: int) -> int:
        n = len(nums)
        S = [x for x in nums if x <= threshold]
        rest_count = n - len(S)
        
        if not S:
            return rest_count
        
        max_val = threshold
        divisors = [[] for _ in range(max_val + 1)]
        
        for x in S:
            m = x
            while m <= max_val:
                divisors[m].append(x)
                m += x
        
        parent = {}
        rank = {}
        for x in S:
            parent[x] = x
            rank[x] = 0
        
        def find(x):
            if parent[x] != x:
                parent[x] = find(parent[x])
            return parent[x]
        
        def union(x, y):
            rx = find(x)
            ry = find(y)
            if rx == ry:
                return
            if rank[rx] < rank[ry]:
                parent[rx] = ry
            elif rank[rx] > rank[ry]:
                parent[ry] = rx
            else:
                parent[ry] = rx
                rank[rx] += 1
        
        for m in range(1, max_val + 1):
            if divisors[m]:
                rep = min(divisors[m])
                for x in divisors[m]:
                    if x != rep:
                        union(rep, x)
        
        roots = set()
        for x in S:
            roots.add(find(x))
        comp_count = len(roots)
        
        return comp_count + rest_count