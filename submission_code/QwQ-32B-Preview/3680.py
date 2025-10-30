from typing import List

class UnionFind:
    def __init__(self, n):
        self.parent = list(range(n))
        self.rank = [0] * n
    def find(self, x):
        if self.parent[x] != x:
            self.parent[x] = self.find(self.parent[x])
        return self.parent[x]
    def union(self, x, y):
        xroot = self.find(x)
        yroot = self.find(y)
        if xroot == yroot:
            return
        if self.rank[xroot] < self.rank[yroot]:
            self.parent[xroot] = yroot
        else:
            self.parent[yroot] = xroot
        if self.rank[xroot] == self.rank[yroot]:
            self.rank[xroot] += 1

class Solution:
    def countComponents(self, nums: List[int], threshold: int) -> int:
        # Separate nums into <= threshold and > threshold
        nums_le_threshold = [num for num in nums if num <= threshold]
        nums_gt_threshold = [num for num in nums if num > threshold]
        n = len(nums_le_threshold)
        if n == 0:
            return len(nums_gt_threshold)
        # Create a map from num to index
        num_to_index = {num: i for i, num in enumerate(nums_le_threshold)}
        # Initialize Union-Find
        uf = UnionFind(n)
        # Mapping from m to list of nums dividing m
        m_to_nums = [[] for _ in range(threshold + 1)]
        # Populate the mapping
        for num in nums_le_threshold:
            multiple = num
            while multiple <= threshold:
                m_to_nums[multiple].append(num)
                multiple += num
        # Connect nums that divide the same m
        for m in range(1, threshold + 1):
            nums_dividing_m = m_to_nums[m]
            for num in nums_dividing_m:
                idx = num_to_index[num]
                # Connect all nums dividing m to the first one in the list
                if m_to_nums[m]:
                    root_idx = num_to_index[m_to_nums[m][0]]
                    uf.union(idx, root_idx)
        # Count the number of unique roots
        roots = set()
        for i in range(n):
            roots.add(uf.find(i))
        # Total components are the unique roots plus the nums > threshold
        total_components = len(roots) + len(nums_gt_threshold)
        return total_components