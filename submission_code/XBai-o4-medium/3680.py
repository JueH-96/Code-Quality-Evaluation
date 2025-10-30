from typing import List

class Solution:
    def countComponents(self, nums: List[int], threshold: int) -> int:
        class DSU:
            def __init__(self, size):
                self.parent = list(range(size))
                self.rank = [1] * size
            def find(self, x):
                if self.parent[x] != x:
                    self.parent[x] = self.find(self.parent[x])
                return self.parent[x]
            def union(self, x, y):
                x_root = self.find(x)
                y_root = self.find(y)
                if x_root == y_root:
                    return
                if self.rank[x_root] < self.rank[y_root]:
                    self.parent[x_root] = y_root
                else:
                    self.parent[y_root] = x_root
                    if self.rank[x_root] == self.rank[y_root]:
                        self.rank[x_root] += 1
        
        small_nums = [x for x in nums if x <= threshold]
        large_count = len(nums) - len(small_nums)
        if not small_nums:
            return large_count
        
        value_to_index = {x: i for i, x in enumerate(small_nums)}
        n = len(small_nums)
        dsu = DSU(n)
        
        max_threshold = threshold
        divisors_list = [[] for _ in range(max_threshold + 1)]
        
        for x in small_nums:
            for m in range(x, max_threshold + 1, x):
                divisors_list[m].append(x)
        
        for m in range(1, max_threshold + 1):
            divisors = divisors_list[m]
            if len(divisors) >= 2:
                first_val = divisors[0]
                first_idx = value_to_index[first_val]
                for i in range(1, len(divisors)):
                    current_val = divisors[i]
                    current_idx = value_to_index[current_val]
                    dsu.union(first_idx, current_idx)
        
        roots = set()
        for i in range(len(small_nums)):
            roots.add(dsu.find(i))
        
        return len(roots) + large_count