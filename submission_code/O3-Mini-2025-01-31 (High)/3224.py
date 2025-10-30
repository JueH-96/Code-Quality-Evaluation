from typing import List

MOD = 10**9 + 7

class Solution:
    def numberOfSequence(self, n: int, sick: List[int]) -> int:
        # Total number of children that are not initially infected.
        H = n - len(sick)
        
        # We'll split the queue into segments of consecutive healthy children.
        # There are three types:
        #  1. A left edge segment (if sick[0] > 0)
        #  2. A right edge segment (if sick[-1] < n-1)
        #  3. Internal segments between consecutive sick positions.
        # 
        # For interleaving the infection events across segments, the count is given by:
        #   (H)! / (prod(segment_length!))
        #
        # In an edge segment there is no choice, because infection can spread only one way.
        # In an internal segment (i.e. flanked by two initially infected children),
        # the infection can spread from both ends. In fact, for an internal segment of length L,
        # there are 2^(L - 1) valid orders (if L >= 1). (For L = 1, 2^(0) = 1.)
        #
        # Thus, the total number is:
        #   interleaving_factor * (∏_{internal segment with length L} [2^(L-1)] )
        
        segments = []   # Will store the lengths of all segments (edge and internal)
        internal_factor = 1  # Product over the 2^(L-1) for every internal (two‐sided) segment
        
        # Left edge segment.
        if sick[0] > 0:
            segments.append(sick[0])  # healthy children from position 0 to sick[0]-1
        
        # Internal segments (between consecutive sick children)
        for i in range(len(sick) - 1):
            gap = sick[i+1] - sick[i] - 1
            if gap > 0:
                segments.append(gap)
                # This segment is flanked by infected children:
                # number of valid orders = 2^(gap-1)
                internal_factor = (internal_factor * pow(2, gap - 1, MOD)) % MOD
        
        # Right edge segment.
        if sick[-1] < n - 1:
            segments.append(n - 1 - sick[-1])
        
        # To count interleavings: the healthy events come from different segments.
        # The number of ways to interleave sequences of lengths l1, l2, ... , lk is:
        #       (l1+l2+...+lk)! / (l1! * l2! * ... * lk!)
        # Here the sum is H.
        # So we need to compute factorials and their modular inverses up to H.
        
        # Precompute factorials and inverse factorials modulo MOD.
        fact = [1] * (H + 1)
        invfact = [1] * (H + 1)
        for i in range(1, H + 1):
            fact[i] = (fact[i-1] * i) % MOD
        invfact[H] = pow(fact[H], MOD - 2, MOD)
        for i in range(H - 1, -1, -1):
            invfact[i] = (invfact[i+1] * (i+1)) % MOD
        
        interleaving = fact[H]
        for seg in segments:
            interleaving = (interleaving * invfact[seg]) % MOD
        
        # Final result is the interleaving multiplied by the internal segments' extra freedom.
        return (interleaving * internal_factor) % MOD

# ---------------------------
# For local quick testing:
if __name__ == "__main__":
    sol = Solution()
    # Example 1:
    n = 5; sick = [0, 4]
    print(sol.numberOfSequence(n, sick))  # Expected output: 4

    # Example 2:
    n = 4; sick = [1]
    print(sol.numberOfSequence(n, sick))  # Expected output: 3