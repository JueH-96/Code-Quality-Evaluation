from typing import List

class Solution:
    def countComponents(self, nums: List[int], threshold: int) -> int:
        # A Disjoint Set Union (Union-Find) implementation
        class DSU:
            def __init__(self, n):
                self.parent = list(range(n))
                self.rank = [0]*n
            
            def find(self, x):
                if self.parent[x] != x:
                    self.parent[x] = self.find(self.parent[x])
                return self.parent[x]
            
            def union(self, x, y):
                rx, ry = self.find(x), self.find(y)
                if rx != ry:
                    if self.rank[rx] < self.rank[ry]:
                        rx, ry = ry, rx
                    self.parent[ry] = rx
                    if self.rank[rx] == self.rank[ry]:
                        self.rank[rx] += 1
        
        # Separate the numbers into those that are <= threshold (A)
        # and those that are > threshold (B). Every number in B
        # cannot connect to any other (since LCM would exceed threshold),
        # so each contributes 1 to the component count by itself.
        A = []
        count_bigger = 0
        for num in nums:
            if num > threshold:
                count_bigger += 1
            else:
                A.append(num)
        
        # If A is empty, all numbers are > threshold and each is isolated
        if not A:
            return count_bigger
        
        # Build a DSU for indices of A
        m = len(A)
        dsu = DSU(m)
        
        # Map each value in A to its index in A
        # (all elements in A are <= threshold)
        index_map = {}
        for i, val in enumerate(A):
            index_map[val] = i
        
        # For each value in A, add it to buckets of its multiples
        # up to 'threshold'. This "multipleMap[k]" will store the indices
        # of elements in A that divide k (i.e., A[i] | k).
        multipleMap = [[] for _ in range(threshold+1)]
        for i, val in enumerate(A):
            step = val
            for multiple in range(step, threshold+1, step):
                multipleMap[multiple].append(i)
        
        # Union all elements that share the same multiple k
        for k in range(1, threshold+1):
            indices = multipleMap[k]
            if len(indices) > 1:
                first = indices[0]
                for j in range(1, len(indices)):
                    dsu.union(first, indices[j])
        
        # Count distinct connected components among A
        roots = set()
        for i in range(m):
            roots.add(dsu.find(i))
        count_a = len(roots)
        
        # Total = components among A plus each number in B as its own component
        return count_a + count_bigger