import sys
sys.setrecursionlimit(300000)

class Solution:
    def resultsArray(self, queries: List[List[int]], k: int) -> List[int]:
        dists = []
        for x, y in queries:
            dists.append(abs(x) + abs(y))
        
        unique_dists = sorted(set(dists))
        comp_map = {}
        for idx, d in enumerate(unique_dists):
            comp_map[d] = idx
        n_original = len(unique_dists)
        
        if n_original == 0:
            return [-1] * len(queries)
        
        n_base = 1
        while n_base < n_original:
            n_base *= 2
        
        tree = [0] * (2 * n_base)
        
        def update(i, delta):
            pos = i + n_base
            tree[pos] += delta
            while pos > 1:
                pos //= 2
                tree[pos] = tree[2*pos] + tree[2*pos+1]
        
        def query_kth(k):
            node = 1
            while node < n_base:
                left_child = 2 * node
                if tree[left_child] >= k:
                    node = left_child
                else:
                    k -= tree[left_child]
                    node = left_child + 1
            return node - n_base
        
        results = []
        total_obstacles = 0
        for i, (x, y) in enumerate(queries):
            d = abs(x) + abs(y)
            idx = comp_map[d]
            update(idx, 1)
            total_obstacles += 1
            if total_obstacles < k:
                results.append(-1)
            else:
                bucket_idx = query_kth(k)
                results.append(unique_dists[bucket_idx])
        
        return results