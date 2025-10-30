from typing import List

MOD = 10**9 + 7

class Solution:
    def numberOfSequence(self, n: int, sick: List[int]) -> int:
        # The idea:
        # Initially infected children are at the positions given in sick.
        # The healthy children are in contiguous segments between and at the endpoints.
        # There are two types of segments:
        #   • One‐sided segments (at the left end before the first sick, or the right end after the last sick).
        #     These segments can only be infected “from one side” so their infection order is forced ("linear").
        #   • Middle segments (gaps between two already infected children).
        #     Their infection can grow from both ends. For a gap with m healthy children,
        #     one can show by a quick check that the number of ways to choose the infection order
        #     within the gap is 2^(m-1) (when m>=1). For m = 1, note that the only child can be infected
        #     from either side, but it is the same child so there is 1 sequence.
        #
        # However, the infection orders of different segments can be interleaved arbitrarily.
        # Suppose there are k segments with lengths L1, L2, ..., Lk (note the one-sided segments
        # have exactly one way to become infected). The ways to interleave the segments is the multinomial
        # coefficient: (T)! / (L1! * L2! * ... * Lk!) where T = sum(Li).
        #
        # Finally, the total number of sequences is:
        #    interleaving ways * (product of [2^(m-1)] for each middle (two‐sided) segment)
        
        segments = []        # list to store the lengths of contiguous healthy segments.
        twoSidedProduct = 1  # product factor from the two-sided (middle) segments
        
        # Left endpoint segment: positions 0 to sick[0]-1
        if sick[0] > 0:
            segments.append(sick[0])
        # Right endpoint segment: positions sick[-1]+1 to n-1
        if sick[-1] < n - 1:
            segments.append(n - 1 - sick[-1])
            
        # Internal segments: gaps between consecutive infected children.
        # They are "two-sided" so the infection can come from both ends.
        for i in range(len(sick) - 1):
            gap = sick[i+1] - sick[i] - 1
            if gap > 0:
                segments.append(gap)
                twoSidedProduct = (twoSidedProduct * pow(2, gap - 1, MOD)) % MOD
                
        # Total number of initially healthy children.
        T = sum(segments)
        
        # To count the number of ways to interleave the infections in different segments, we want:
        #    ways = T! / (L1! * L2! * ... * Lk!)
        #
        # We will precompute factorials and their inverses modulo MOD.
        fact = [1] * (T + 1)
        invfact = [1] * (T + 1)
        for i in range(2, T + 1):
            fact[i] = fact[i - 1] * i % MOD
            
        invfact[T] = pow(fact[T], MOD - 2, MOD)
        for i in range(T, 0, -1):
            invfact[i - 1] = invfact[i] * i % MOD
        
        interleavings = fact[T]
        for length in segments:
            interleavings = (interleavings * invfact[length]) % MOD
        
        return interleavings * twoSidedProduct % MOD

# The following is a main routine to facilitate testing. 
# It is not required if you are integrating the class into a different environment.
if __name__ == '__main__':
    import sys
    input_data = sys.stdin.read().strip().split()
    if input_data:
        # The expected input: first integer n, followed by the list of sick positions.
        # For example, input might be: "5 0 4" corresponding to n = 5, sick = [0, 4].
        n = int(input_data[0])
        sick = list(map(int, input_data[1:]))
        sol = Solution()
        result = sol.numberOfSequence(n, sick)
        sys.stdout.write(str(result))