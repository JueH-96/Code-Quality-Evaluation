from typing import List
import math

class UnionFind:
    def __init__(self, n):
        self.parent = list(range(n))
        self.rank = [0] * n

    def find(self, x):
        if self.parent[x] != x:
            self.parent[x] = self.find(self.parent[x])
        return self.parent[x]
    
    def union(self, x, y):
        xr = self.find(x)
        yr = self.find(y)
        if xr == yr:
            return False
        if self.rank[xr] < self.rank[yr]:
            self.parent[xr] = yr
        elif self.rank[xr] > self.rank[yr]:
            self.parent[yr] = xr
        else:
            self.parent[yr] = xr
            self.rank[xr] += 1
        return True

class Solution:
    def countComponents(self, nums: List[int], threshold: int) -> int:
        n = len(nums)
        # Nodes with value greater than threshold have no edge: because
        # if a > threshold then any lcm(a, b) >= a > threshold.
        isolated_count = 0
        valid = []  # (value, original index) for values <= threshold
        for i, num in enumerate(nums):
            if num > threshold:
                isolated_count += 1
            else:
                valid.append((num, i))
        
        # If no valid nodes, answer is just number of isolated nodes.
        if not valid:
            return isolated_count
        
        # Build mapping: for each number x in valid group, we'll store its index (in union-find array)
        # We'll use union-find on the valid nodes.
        m = len(valid)
        uf = UnionFind(m)
        
        # Create a list of lists (bucket) for each integer d from 1 to threshold.
        # For each d, we will list all valid nodes (by their union-find id) that divide d.
        # A common d means there exists a multiple d (<= threshold) that both numbers divide.
        # Then lcm(a,b) divides this d, implying lcm(a,b) <= d <= threshold.
        buckets = [[] for _ in range(threshold + 1)]
        
        # We iterate over each valid number and mark its multiples in the range [num, threshold].
        for uf_id, (val, orig_index) in enumerate(valid):
            # For multiples m_val of val from val to threshold
            step = val
            for d in range(val, threshold + 1, step):
                buckets[d].append(uf_id)
        
        # For every d in [1, threshold] if there are multiple numbers that divide d, union them.
        for d in range(1, threshold + 1):
            if len(buckets[d]) > 1:
                base = buckets[d][0]
                for other in buckets[d][1:]:
                    uf.union(base, other)
        
        # Count distinct connected groups in valid group
        unique_parents = set()
        for i in range(m):
            unique_parents.add(uf.find(i))
        valid_components = len(unique_parents)
        
        # Total connected components = connected group from valid + isolated nodes (> threshold)
        return valid_components + isolated_count


# Example to test the implementation
if __name__ == '__main__':
    sol = Solution()
    print(sol.countComponents([2,4,8,3,9], 5))    # Expected output: 4
    print(sol.countComponents([2,4,8,3,9,12], 10))  # Expected output: 2