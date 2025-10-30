class Solution:
    def minOperations(self, k: int) -> int:
        if k == 1:
            return 0
        min_ops = float('inf')
        # We try to find the optimal number of duplications
        # Let's say we duplicate the initial 1 'd' times
        # Then, the array will have (d+1) elements, each initially 1
        # We need to increase some of them to make the sum >= k
        # The optimal way is to increase all elements equally
        # So, each element will be increased by x times
        # The sum will be (d+1) * (1 + x) >= k
        # The total operations will be d (for duplications) + (d+1)*x (for increments)
        # We need to minimize d + (d+1)*x
        # Since x >= 0, and (d+1)*(1 + x) >= k
        # x >= (k / (d+1)) - 1
        # So, x = ceil((k / (d+1)) - 1)
        # We can iterate over possible d and compute the minimal operations
        # Since d can be up to k-1 (but in practice, much less)
        # We can limit d to sqrt(k) for efficiency
        max_d = int((k) ** 0.5) + 2
        for d in range(0, max_d):
            x = max(0, (k + (d)) // (d + 1) - 1)
            total_ops = d + (d + 1) * x
            if (d + 1) * (1 + x) >= k:
                if total_ops < min_ops:
                    min_ops = total_ops
        return min_ops