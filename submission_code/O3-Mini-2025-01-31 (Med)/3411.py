from typing import List
import math

class Solution:
    def findProductsOfElements(self, queries: List[List[int]]) -> List[int]:
        # We note that each element in big_nums is a power-of-two: 2^k.
        # The product of any subsequence is 2^(sum of k’s in that subsequence) mod mod.
        # Hence the problem reduces to: for each query on indices [L, R] (0-indexed)
        # in the concatenated sequence big_nums – which is the concatenation of each positive integer’s “powerful array”
        # (its binary representation written as sorted powers of 2, i.e. for i, if bit k is set, then its contribution is 2^k) –
        # we need to compute the sum of exponents in that segment.
        #
        # Let pop(i) = popcount(i) = number of 1 bits in i.
        # Then the length of the "powerful array" for integer i is pop(i).
        # Define S(n) = sum_{i=1}^n pop(i) which is the total length of big_nums for integers 1..n.
        # Also, for each integer i, let g(i) = sum_{k: (i>>k)&1==1} k   (i.e. sum of indices 
        # where the bit is 1). Then the product for i’s powerful array is 2^(g(i)).
        # Define W(n) = sum_{i=1}^n g(i).
        #
        # Then the prefix (in big_nums) contribution (i.e. sum of exponents) for a “complete block”
        # corresponding to integers 1..n is exactly W(n).
        #
        # But a query [L,R] may start or end in the middle of the “powerful array” for some integer.
        # So we need to be able to “invert” S(n). Given a 0-indexed position pos in big_nums,
        # we want to find the integer r such that S(r-1) <= pos < S(r). Then the prefix sum up to pos is:
        #     prefix(pos) = W(r-1) + partial(r, offset)
        # where offset = pos - S(r-1) + 1   (number of entries taken from integer r)
        # and partial(r, offset) is the sum of the exponents from the first offset entries in r's sorted list.
        # In r’s binary representation the bits (if set) sorted in increasing order correspond to exponents: 
        # e.g. r = 11 (binary 1011) has its set bits at indices [0, 1, 3] (in sorted order).
        #
        # To answer a query, one computes:
        #   expo = prefix(R) - (prefix(L-1) if L > 0 else 0)
        # and the answer is: 2^(expo) mod mod.
        #
        # Because the queries can have indices as high as 1e15, we have to invert S(n) by binary search.
        # We need efficient formulas for S(n) and for W(n).
        #
        # It is known that the sum of popcounts from 1 to n follows a recurrence.
        # Let f(n) = S(n):
        #   if n==0: return 0.
        #   Let p = floor(log2(n)) and power = 2^p.
        #   Then f(n) = p * (power >> 1) + (n - power + 1) + f(n - power).
        #
        # Similarly, let F(n) = W(n) = sum_{i=1}^n g(i), where g(i) = sum of bit-indices for bits that are set in i.
        # For n = 0: F(0)=0.
        # For n>0, let p = floor(log2(n)), power = 2^p.
        # For numbers 1 to power-1, i.e. for n = 2^p - 1, every bit position j in [0, p-1] appears exactly 2^(p-1) times.
        # So the sum for these numbers is: sum_{j=0}^{p-1} j * 2^(p-1) = (2**(p-1)) * (p*(p-1)//2).
        # Then for n in [power, n]:
        #    Their most significant bit (p) is set in all (n - power + 1) numbers; so add p * (n - power + 1),
        #    plus add F(n - power) recursively.
        #
        # Hence, the recurrence for F(n) is:
        #   F(n) = ( (power >> 1) * (p*(p-1)//2) ) + (n - power + 1) * p + F(n - power).
        #
        # We implement these recurrences (which run in O(bit_length(n)) steps).
        #
        # Finally, we use binary search to invert S(n). Given pos (0-indexed in big_nums),
        # we want the smallest integer r such that S(r) > pos.
        #
        # For computing the partial sum from an integer r’s "powerful array" we do:
        #   Let bits = sorted([k for k in range(r.bit_length()) if ((r >> k) & 1) == 1]).
        #   Then partial(r, m) = sum(bits[0:m])  (m is guaranteed to be <= len(bits)).
        #
        # Now, answer each query.
        
        # --- helper functions for S(n) and F(n) using recursion ---
        def sumPop(n: int) -> int:
            # Returns S(n) = sum_{i=1}^n popcount(i)
            if n <= 0:
                return 0
            # p = floor(log2(n))
            p = n.bit_length() - 1
            power = 1 << p  # 2^p
            # For numbers 1 ... 2^p - 1, the sum of popcounts is: p * 2^(p-1)
            return p * (power >> 1) + (n - power + 1) + sumPop(n - power)
        
        def sumWeighted(n: int) -> int:
            # Returns F(n) = sum_{i=1}^n g(i) where g(i) = sum of indices (exponents) for each set bit in i.
            if n <= 0:
                return 0
            p = n.bit_length() - 1
            power = 1 << p
            # For numbers 1..(2^p - 1):
            # each bit position j in 0...(p-1) appears exactly 2^(p-1) times,
            # so contribution = sum_{j=0}^{p-1} j * 2^(p-1) = 2^(p-1) * (p*(p-1)//2)
            base = (power >> 1) * (p * (p - 1) // 2)
            return base + (n - power + 1) * p + sumWeighted(n - power)
        
        # --- function to compute partial contribution for a single number r ---
        def partialContribution(r: int, m: int) -> int:
            # r's "powerful array" is just the sorted list (in increasing order) of exponents k for which (r >> k) & 1 == 1.
            bits = []
            bl = r.bit_length()
            for k in range(bl):
                if (r >> k) & 1:
                    bits.append(k)
            # bits list is already in increasing order because k runs from 0 upward.
            # m is between 1 and len(bits)
            return sum(bits[:m])
        
        # --- function to compute prefix sum of exponents in big_nums up to index pos (0-indexed) ---
        def prefixExponentSum(pos: int) -> int:
            # Find integer r such that sumPop(r-1) <= pos < sumPop(r)
            # We'll do binary search over r.
            lo, hi = 1, pos + 1  # since S(n) >= n, r will be at most pos+1.
            # Increase hi until we are sure that sumPop(hi) > pos.
            while sumPop(hi) <= pos:
                hi *= 2
            while lo < hi:
                mid = (lo + hi) // 2
                if sumPop(mid) > pos:
                    hi = mid
                else:
                    lo = mid + 1
            r = lo
            # Sum for all complete numbers before r:
            prev_full = sumPop(r - 1) if r > 1 else 0
            exp_before = sumWeighted(r - 1) if r > 1 else 0
            offset = pos - prev_full + 1  # number of entries from r's block to include
            # Now compute partial contribution from r.
            partial_val = partialContribution(r, offset)
            return exp_before + partial_val
        
        # Process each query.
        ans = []
        for L, R, mod in queries:
            # Compute exponent sum in segment = prefix(R) - prefix(L-1) (with prefix(-1)=0)
            expo_R = prefixExponentSum(R)
            expo_before = prefixExponentSum(L - 1) if L > 0 else 0
            expo = expo_R - expo_before
            # The product is 2^(expo) mod mod.
            res = pow(2, expo, mod)
            ans.append(res)
        return ans