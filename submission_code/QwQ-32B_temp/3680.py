class Solution:
    def countComponents(self, nums: List[int], threshold: int) -> int:
        subset = []
        others = 0
        elements_set = set()
        for num in nums:
            if num <= threshold:
                subset.append(num)
                elements_set.add(num)
            else:
                others += 1
        if not subset:
            return others
        
        # Precompute divisors list
        T = threshold
        divisors_list = [[] for _ in range(T + 1)]
        for d in range(1, T + 1):
            for multiple in range(d, T + 1, d):
                divisors_list[multiple].append(d)
        
        # Initialize DSU (Union-Find)
        parent = {}
        rank = {}
        for num in subset:
            parent[num] = num
            rank[num] = 1
        
        def find(x):
            if parent[x] != x:
                parent[x] = find(parent[x])
            return parent[x]
        
        def union(x, y):
            x_root = find(x)
            y_root = find(y)
            if x_root == y_root:
                return
            if rank[x_root] < rank[y_root]:
                parent[x_root] = y_root
            else:
                parent[y_root] = x_root
                if rank[x_root] == rank[y_root]:
                    rank[x_root] += 1
        
        for c in range(1, T + 1):
            divisors = []
            for d in divisors_list[c]:
                if d in elements_set:
                    divisors.append(d)
            if len(divisors) >= 2:
                first = divisors[0]
                for d in divisors[1:]:
                    union(first, d)
        
        # Count the number of unique roots
        roots = set()
        for num in subset:
            roots.add(find(num))
        
        return len(roots) + others