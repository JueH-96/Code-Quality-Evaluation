class Solution:
    def minOperations(self, k: int) -> int:
        # We start with nums = [1]. We can:
        #  - increment any element by 1 (cost 1)
        #  - duplicate any element (cost 1)
        # We want sum(nums) >= k at minimal total ops.
        # Optimal strategy: pick a final value m for all elements,
        # then have array size (t+1), all equal m, so sum = m*(t+1) >= k.
        # Cost = (m-1) increments on the original 1 -> m
        #      + t duplications = (m-1) + t
        # We need t+1 >= ceil(k/m) => t >= ceil(k/m)-1.
        # So for a given m, cost = (m-1) + (ceil(k/m)-1) = m + ceil(k/m) -2.
        # We minimize over m >= 1.
        # We search over small m (up to sqrt) and small t = ceil(k/m)-1 likewise.
        
        if k <= 1:
            return 0
        
        import math
        N = k
        ans = float('inf')
        # search over m up to sqrt(N)
        limit = int(math.sqrt(N)) + 2
        for m in range(1, limit):
            # minimal t so that (t+1)*m >= N
            t = (N + m - 1) // m - 1
            ops = (m - 1) + t
            if ops < ans:
                ans = ops
        
        # also search over small t up to sqrt(N)
        # here t is number of duplicates, so array size = t+1
        for t in range(0, limit):
            # minimal m so that (t+1)*m >= N
            size = t + 1
            m = (N + size - 1) // size
            ops = (m - 1) + t
            if ops < ans:
                ans = ops
        
        return ans