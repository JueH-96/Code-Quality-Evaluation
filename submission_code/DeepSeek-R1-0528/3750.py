from collections import defaultdict

class Solution:
    def solveQueries(self, nums: list, queries: list) -> list:
        n = len(nums)
        groups = defaultdict(list)
        for i, num in enumerate(nums):
            groups[num].append(i)
        
        res = [-1] * n
        
        for key, indices in groups.items():
            m = len(indices)
            if m == 1:
                continue
            for k in range(m):
                current = indices[k]
                next_node = indices[(k + 1) % m]
                prev_node = indices[(k - 1) % m]
                d_next = min(abs(current - next_node), n - abs(current - next_node))
                d_prev = min(abs(current - prev_node), n - abs(current - prev_node))
                res[current] = min(d_next, d_prev)
        
        return [res[q] for q in queries]