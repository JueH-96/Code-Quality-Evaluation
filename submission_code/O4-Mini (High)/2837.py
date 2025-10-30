class Solution:
    def makeTheIntegerZero(self, num1: int, num2: int) -> int:
        # If num2 == 0, each operation subtracts a pure power of two.
        # We need exactly popcount(num1) such operations.
        if num2 == 0:
            return bin(num1).count("1")
        
        # For num2 > 0, k is bounded by num1 // (num2+1) via the N >= k constraint.
        # For num2 < 0, there's no upper bound on k from that constraint.
        if num2 > 0:
            maxK = num1 // (num2 + 1)
        else:
            maxK = float('inf')
        
        # We only need to try k up to a small constant (≈60) because for larger k
        # the popcount constraint bin(N).count('1') <= k is automatically satisfied.
        for k in range(1, 64):
            if k > maxK:
                break
            N = num1 - k * num2
            # We must have N = sum of k powers of two, so N >= k
            if N < k:
                continue
            # And popcount(N) <= k is both necessary and sufficient for such a representation
            if bin(N).count("1") <= k:
                return k
        
        return -1