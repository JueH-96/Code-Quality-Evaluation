from math import gcd
from collections import defaultdict
from typing import List

class DSU:
    def __init__(self, n: int):
        self.parent = list(range(n))
        
    def find(self, a: int) -> int:
        while self.parent[a] != a:
            self.parent[a] = self.parent[self.parent[a]]
            a = self.parent[a]
        return a
    
    def union(self, a: int, b: int) -> None:
        pa = self.find(a)
        pb = self.find(b)
        if pa != pb:
            self.parent[pb] = pa

class Solution:
    def countComponents(self, nums: List[int], threshold: int) -> int:
        # We interpret the graph as follows:
        # Two nodes (with values a and b) are connected by an edge if lcm(a, b) <= threshold.
        # Note that lcm(a, b) = a * b // gcd(a, b).
        #
        # A key observation is that if either a or b is greater than threshold,
        # then lcm(a,b) >= max(a,b) > threshold. So such nodes CANNOT connect with any other node.
        #
        # Therefore, we can separate the nodes into:
        #   - "Small" nodes: those with value <= threshold.
        #   - "Large" nodes: those with value > threshold.
        # Large nodes are isolated.
        #
        # Next, for the small nodes we need to determine connectivity.
        # Notice that for two small numbers, a,b <= threshold, the condition lcm(a,b) <= threshold
        # is equivalent to: a * b // gcd(a,b) <= threshold.
        # Checking every pair among small nodes is too slow.
        #
        # Instead, we use the following idea:
        # For any common number L (1 <= L <= threshold), if two different small numbers both
        # divide L then L is a common multiple of the two numbers. In particular, since the minimal
        # common multiple is lcm(a,b), we have lcm(a,b) <= L <= threshold.
        # So if for some L, at least two small numbers divide L then these numbers must be connected.
        #
        # We then use a sieve–like method.
        #
        # Steps:
        # 1. Partition the numbers: S (small numbers <= threshold) and isolated (large numbers > threshold).
        # 2. Build a mapping from value to its index in an array "arr" representing S.
        # 3. For each small number d, iterate over its multiples m (from d to threshold, step d)
        #    and for m, record that d (its corresponding index) divides m.
        # 4. For each m from 1 to threshold, if more than one small number divides m,
        #    union them in DSU.
        # 5. The connected components among small numbers are given by DSU,
        #    and each large number is an isolated component.
        
        # Separate small and large nodes.
        small = []
        small_indices = {}  # mapping: value -> index in "small" list
        large_count = 0
        for num in nums:
            if num <= threshold:
                small_indices[num] = len(small)
                small.append(num)
            else:
                large_count += 1
        
        # DSU for the small nodes.
        dsu = DSU(len(small))
        
        # For each L in [1, threshold], we want to collect indices of small numbers that divide L.
        # We build an array groups where groups[m] is the list of indices in "small" of numbers dividing m.
        groups = [[] for _ in range(threshold + 1)]
        for d in small:
            idx = small_indices[d]
            # Iterate multiples m of d up to threshold.
            for m in range(d, threshold+1, d):
                groups[m].append(idx)
        
        # For each multiple m, union all indices in groups[m]
        for group in groups:
            if len(group) > 1:
                base = group[0]
                for other in group[1:]:
                    dsu.union(base, other)
        
        # Count number of connected components among small nodes.
        comp_set = set()
        for i in range(len(small)):
            comp_set.add(dsu.find(i))
        small_components = len(comp_set)
        
        # The answer is the sum of connected components among small nodes and
        # the isolated large nodes.
        return small_components + large_count