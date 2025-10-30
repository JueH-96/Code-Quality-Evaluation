from typing import List

class Solution:
    def countComponents(self, nums: List[int], threshold: int) -> int:
        M = threshold
        # Map small values (<=M) to indices, count how many are >M (they are isolated)
        present = [-1] * (M + 1)
        small = []
        big_count = 0
        for x in nums:
            if x <= M:
                present[x] = len(small)
                small.append(x)
            else:
                big_count += 1

        n0 = len(small)
        # If no small values, each big value is its own component
        if n0 == 0:
            return big_count

        # Disjoint‐set (union‐find) for small values
        parent = list(range(n0))
        size = [1] * n0
        comp_count = n0

        def find(x):
            # Path compression
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x

        def union(a, b):
            nonlocal comp_count
            ra = find(a)
            rb = find(b)
            if ra == rb:
                return
            # Union by size
            if size[ra] < size[rb]:
                ra, rb = rb, ra
            parent[rb] = ra
            size[ra] += size[rb]
            comp_count -= 1

        # Build buckets: for each ℓ ≤ M, keep indices of small values that divide ℓ
        buckets = [[] for _ in range(M + 1)]
        for idx, x in enumerate(small):
            step = x
            for ℓ in range(step, M + 1, step):
                buckets[ℓ].append(idx)

        # For each ℓ, all values dividing ℓ have pairwise lcm ≤ ℓ ≤ M, so union them
        for bucket in buckets[1:]:
            if len(bucket) > 1:
                first = bucket[0]
                for idx in bucket[1:]:
                    union(first, idx)

        # Total components = components among small + each big value on its own
        return comp_count + big_count