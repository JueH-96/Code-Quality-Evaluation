from typing import List

class Solution:
    def maximumSubarrayXor(self, nums: List[int], queries: List[List[int]]) -> List[int]:
        n = len(nums)
        # Precompute XOR prefix: p[i] = nums[0]⊕nums[1]⊕…⊕nums[i-1], with p[0] = 0.
        p = [0]*(n+1)
        for i in range(n):
            p[i+1] = p[i] ^ nums[i]
        
        # For any even length L (>=2) we want to “correct” the XOR of the whole block.
        # Define:
        #   S = set of indices k in [0,L) that are used.
        #   (By Lucas theorem, S = { k: k is a submask of L-1 } )
        # Then:
        #   f(i, j) = (p[i+L] ^ p[i])  ⊕ (XOR over r in M),
        # where M = [0,L) \ S.
        # It turns out that M (for given even L) is a union of a few contiguous intervals.
        # We precompute, for even L, the list of intervals (start, end) (inclusive) in [0,L)
        # that are NOT allowed (i.e. that are not submasks of L-1).
        
        ev_intervals = {}
        # Only even L >=2 occur because subarray of length 1 is odd.
        for L in range(2, n+1, 2):
            t = L - 1
            # Represent t in minimal bits.
            b = t.bit_length() if t != 0 else 1
            full_mask = (1 << b) - 1
            # In the b–bit world, the allowed numbers are the submasks of t.
            # So a number r (0 ≤ r < L) is allowed if and only if (r & forbidden)==0,
            # where forbidden = full_mask ^ t (the bits 0 in t).
            forbidden = full_mask ^ t
            not_allowed = []
            for r in range(L):
                if r & forbidden:
                    not_allowed.append(r)
            # now group not_allowed into contiguous intervals
            intervals = []
            if not_allowed:
                start = not_allowed[0]
                prev = start
                for r in not_allowed[1:]:
                    if r == prev + 1:
                        prev = r
                    else:
                        intervals.append((start, prev))
                        start = r
                        prev = r
                intervals.append((start, prev))
            ev_intervals[L] = intervals
        # End precomputation of even-length intervals
        
        # Build dp table:
        # We use a “compressed” dp so that for each starting index i, we store a list dp[i]
        # of length (n-i) where dp[i][j-i] = maximum XOR score for any subarray in nums[i..j].
        dp = [ [0]*(n-i) for i in range(n) ]
        
        # Fill dp in order: for i from n-1 downto 0, and for each j starting at i.
        for i in range(n-1, -1, -1):
            # subarray [i,i] has score = nums[i]
            dp[i][0] = nums[i]
            # diff = j - i  (so length L = diff+1)
            for diff in range(1, n - i):
                j = i + diff
                L = diff + 1  # length of subarray nums[i..j]
                # For odd lengths: for L>=3, the final score is nums[i]⊕nums[j]
                if L & 1:  # odd length
                    # (length 1 already dealt with)
                    f_val = nums[i] ^ nums[j]
                else:
                    # even length:
                    # Let A be the XOR of all elements in nums[i..j]:
                    A = p[i+L] ^ p[i]
                    # For even lengths not of the form 2^k, the correction factor is nonzero.
                    Corr = 0
                    for s, e in ev_intervals[L]:
                        # The XOR of nums[i+s..i+e] is  p[i+e+1] ^ p[i+s]
                        Corr ^= (p[i+e+1] ^ p[i+s])
                    f_val = A ^ Corr
                
                # dp[i][j-i] = max( f(i,j), dp[i][j-i-1] (best in [i, j-1]), dp[i+1][j-i-1] (best in [i+1,j]) )
                best_val = f_val
                # what is best for subarray [i, j-1]? (provided diff>=1)
                if dp[i][diff-1] > best_val:
                    best_val = dp[i][diff-1]
                if i+1 < n and dp[i+1][diff-1] > best_val:
                    best_val = dp[i+1][diff-1]
                dp[i][diff] = best_val
        # End dp precomputation.
        
        # Answer queries: answer for query [L,R] is dp[L][R-L]
        ans = [ dp[L][R-L] for L,R in queries ]
        return ans