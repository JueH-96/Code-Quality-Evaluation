class Solution:
    def countComponents(self, nums: List[int], threshold: int) -> int:
        parent = {}
        rank = {}
        
        def find(u):
            if parent[u] != u:
                parent[u] = find(parent[u])
            return parent[u]
        
        def union(u, v):
            u_root = find(u)
            v_root = find(v)
            if u_root == v_root:
                return
            if rank[u_root] < rank[v_root]:
                parent[u_root] = v_root
            else:
                parent[v_root] = u_root
                if rank[u_root] == rank[v_root]:
                    rank[u_root] += 1
        
        # Initialize parent and rank for all elements
        for num in nums:
            parent[num] = num
            rank[num] = 1
        
        small_nums = [x for x in nums if x <= threshold]
        max_m = threshold
        divisors = [[] for _ in range(max_m + 1)]
        
        for x in small_nums:
            m = x
            while m <= max_m:
                divisors[m].append(x)
                m += x
        
        for m in range(1, max_m + 1):
            if len(divisors[m]) >= 1:
                root = divisors[m][0]
                for num in divisors[m][1:]:
                    union(root, num)
        
        roots = set()
        for num in nums:
            roots.add(find(num))
        
        return len(roots)