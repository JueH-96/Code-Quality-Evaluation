class Solution:
    def countComponents(self, nums: List[int], threshold: int) -> int:
        from collections import defaultdict

        class UnionFind:
            def __init__(self, n):
                self.parent = list(range(n))

            def find(self, x):
                while x != self.parent[x]:
                    self.parent[x] = self.parent[self.parent[x]]  # Path compression
                    x = self.parent[x]
                return x

            def union(self, x, y):
                px = self.find(x)
                py = self.find(y)
                if px != py:
                    self.parent[px] = py

        n = len(nums)
        num_to_index = {num: idx for idx, num in enumerate(nums)}

        uf = UnionFind(n)

        max_num = max(nums)
        limit = max(threshold + 1, max_num + 1)

        # Only nums[i] <= threshold can be connected
        # nums[i] > threshold are isolated components
        # So we only process nums[i] <= threshold
        nums_set = set(nums)
        for num in nums:
            if num <= threshold:
                # For multiples of num greater than num and less than or equal to max_num
                multiple = num * 2
                while multiple <= max_num:
                    if multiple in nums_set and multiple <= threshold:
                        uf.union(num_to_index[num], num_to_index[multiple])
                    multiple += num

        # Count the number of connected components
        components = set()
        for i in range(n):
            parent = uf.find(i)
            components.add(parent)

        return len(components)