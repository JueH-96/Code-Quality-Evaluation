class UnionFind:
    def __init__(self, size):
        self.root = [i for i in range(size)]
        self.rank = [1] * size

    def find(self, x):
        if x == self.root[x]:
            return x
        self.root[x] = self.find(self.root[x])
        return self.root[x]

    def union(self, x, y):
        rootX = self.find(x)
        rootY = self.find(y)
        if rootX != rootY:
            if self.rank[rootX] > self.rank[rootY]:
                self.root[rootY] = rootX
            elif self.rank[rootX] < self.rank[rootY]:
                self.root[rootX] = rootY
            else:
                self.root[rootY] = rootX
                self.rank[rootX] += 1
        return

class Solution:
    def countComponents(self, nums: List[int], threshold: int) -> int:
        uf = UnionFind(len(nums))
        factors = [[] for _ in range(len(nums))]
        
        for i, num in enumerate(nums):
            for j in range(1, int(sqrt(threshold)) + 1):
                if num % j == 0:
                    k = j
                    if k * k > num:
                        k = num // j
                    if k * num <= threshold:
                        factors[i].append(k)
            factors[i].append(num)

        visited = set()
        for i in range(len(nums)):
            if nums[i] > threshold:
                continue
            for factor in factors[i]:
                for j in range(len(nums)):
                    if i == j or nums[j] > threshold or j in visited:
                        continue
                    if lcm(nums[i], nums[j]) <= threshold:
                        uf.union(i, j)
                        visited.add(j)
        
        return sum(i == uf.find(i) for i in range(len(nums)))