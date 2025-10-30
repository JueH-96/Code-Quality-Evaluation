from typing import List

class Solution:
    def canMakePalindromeQueries(self, s: str, queries: List[List[int]]) -> List[bool]:
        # Precompute length and split point.
        n = len(s)
        L = n // 2  # left half: indices 0..L-1, right half: indices L..n-1
        
        # Precompute prefix frequency arrays for left and right halves.
        # We'll create two 2D arrays for 26 letters.
        left_prefix = [[0]*(L+1) for _ in range(26)]
        right_prefix = [[0]*(L+1) for _ in range(26)]
        
        for i in range(L):
            ch = ord(s[i]) - 97
            for letter in range(26):
                left_prefix[letter][i+1] = left_prefix[letter][i]
            left_prefix[ch][i+1] += 1
            
        for i in range(L, n):
            ch = ord(s[i]) - 97
            j = i - L  # index in right_prefix array
            for letter in range(26):
                right_prefix[letter][j+1] = right_prefix[letter][j]
            right_prefix[ch][j+1] += 1
            
        # Precompute mismatch prefix for fixed pairs.
        # For each i in [0, L-1], define mismatch[i]=1 if s[i] != s[n-1-i] else 0.
        mismatch = [0]*L
        for i in range(L):
            if s[i] != s[n-1-i]:
                mismatch[i] = 1
        mismatch_prefix = [0]*(L+1)
        for i in range(L):
            mismatch_prefix[i+1] = mismatch_prefix[i] + mismatch[i]
            
        # Helper: get frequency array for left half in range [l, r] (0-indexed in left half)
        def query_left(l: int, r: int) -> List[int]:
            res = [0]*26
            if l > r:
                return res
            for letter in range(26):
                res[letter] = left_prefix[letter][r+1] - left_prefix[letter][l]
            return res

        # Helper: get frequency array for right half in range [l, r] (0-indexed in right half)
        def query_right(l: int, r: int) -> List[int]:
            res = [0]*26
            if l > r:
                return res
            for letter in range(26):
                res[letter] = right_prefix[letter][r+1] - right_prefix[letter][l]
            return res

        # Helper: get sum of mismatches over left indices [l, r]
        def query_mismatch(l: int, r: int) -> int:
            if l > r:
                return 0
            return mismatch_prefix[r+1] - mismatch_prefix[l]

        ans = []
        # Process each query.
        for a, b, c, d in queries:
            possible = True
            # The allowed rearrangement segments are in:
            # left allowed: indices [a, b] (subset of left half, indices 0 to L-1)
            # right allowed: indices [c, d] in the right half.
            # It is more convenient to talk in terms of left-half indices.
            # For an index i in left half, the partner is j = n-1 - i.
            # i is "allowed from right side" if its partner j is in [c,d].
            # Since right half indices start at L, we have: j = 2L-1 - i in [c, d].
            # Solve for i: 2L-1 - i in [c, d]  => i in [2L - 1 - d, 2L - 1 - c].
            Lr_low = 2 * L - 1 - d   # lowest i that gets "right allowed" status.
            Lr_high = 2 * L - 1 - c  # highest i with right allowed.
            # Ensure boundaries lie within [0,L-1]
            if Lr_low < 0: 
                Lr_low = 0
            if Lr_high > L-1:
                Lr_high = L-1
            
            # Define the intervals in left half:
            # A = [a, b] are left indices that we can rearrange.
            # B = [Lr_low, Lr_high] are left indices whose partner is rearrangeable.
            # Then each left index i falls into one of 4 categories:
            # 1) i in A and i in B  => both allowed => free pair.
            # 2) i in A but not in B    => forced left: since left side can change but right partner is fixed.
            # 3) i not in A but in B    => forced right: left fixed, but right side can be changed.
            # 4) i not in A and not in B => fixed pair, must already match.
            
            # Check fixed pairs: i not in A and not in B.
            # The set of fixed indices = [0, L-1] \ (A ∪ B).
            # We can get the sum of mismatches over [0,L-1] and subtract mismatches in A and in B,
            # but note that A and B might overlap, so add back intersection.
            total_mismatch = query_mismatch(0, L-1)
            mismatch_A = query_mismatch(a, b) if a <= b else 0
            mismatch_B = query_mismatch(Lr_low, Lr_high) if Lr_low <= Lr_high else 0
            # Intersection of A and B = [max(a, Lr_low), min(b, Lr_high)]
            inter_low = max(a, Lr_low)
            inter_high = min(b, Lr_high)
            mismatch_inter = query_mismatch(inter_low, inter_high) if inter_low <= inter_high else 0
            fixed_mismatch = total_mismatch - mismatch_A - mismatch_B + mismatch_inter
            if fixed_mismatch != 0:
                ans.append(False)
                continue  # this query fails.
            
            # Next, available letters in allowed positions (pools):
            # Left pool (indices [a, b]) and Right pool (indices [c, d]).
            avail_left = query_left(a, b)
            avail_right = query_right(c - L, d - L)  # convert right half indices to 0-indexed for right_prefix
            
            # Forced left: indices in A \ B.
            forced_left_req = [0]*26
            
            # A \ B can be at most two segments:
            # Segment 1: from a to min(b, Lr_low - 1) if Lr_low > a.
            if a <= b and Lr_low > a:
                seg1_end = min(b, Lr_low - 1)
                # For each index i in [a, seg1_end], partner j = 2L - 1 - i
                # This corresponds to a contiguous segment in right half:
                partner_low = 2 * L - 1 - seg1_end
                partner_high = 2 * L - 1 - a
                # partner indices lie in [partner_low, partner_high] (in right half, 0-indexed offset: subtract L)
                # Check boundaries: They must be within [L, n-1]. They will be because a, seg1_end are in [0, L-1].
                # Convert to right_prefix index: subtract L.
                left_counts = query_right(partner_low - L, partner_high - L)
                for letter in range(26):
                    forced_left_req[letter] += left_counts[letter]
            
            # Segment 2: from max(a, Lr_high+1) to b, if Lr_high < b.
            if a <= b and Lr_high < b:
                seg2_start = max(a, Lr_high + 1)
                seg2_end = b
                partner_low = 2 * L - 1 - seg2_end
                partner_high = 2 * L - 1 - seg2_start
                left_counts = query_right(partner_low - L, partner_high - L)
                for letter in range(26):
                    forced_left_req[letter] += left_counts[letter]
            
            # Forced right: indices in B \ A.
            forced_right_req = [0]*26
            # B \ A: from B = [Lr_low, Lr_high] minus A.
            # Segment 1: [Lr_low, min(Lr_high, a-1)] if a > Lr_low.
            if Lr_low <= Lr_high and a > Lr_low:
                seg1_end = min(Lr_high, a - 1)
                left_counts = query_left(Lr_low, seg1_end)
                for letter in range(26):
                    forced_right_req[letter] += left_counts[letter]
            # Segment 2: [max(Lr_low, b+1), Lr_high] if b < Lr_high.
            if Lr_low <= Lr_high and b < Lr_high:
                seg2_start = max(Lr_low, b + 1)
                left_counts = query_left(seg2_start, Lr_high)
                for letter in range(26):
                    forced_right_req[letter] += left_counts[letter]
            
            # Now, we must have that the forced letter requirements
            # can be met from the available pools:
            valid = True
            for letter in range(26):
                if avail_left[letter] < forced_left_req[letter]:
                    valid = False
                    break
                if avail_right[letter] < forced_right_req[letter]:
                    valid = False
                    break
                # After satisfying forced assignments, the leftovers in left and right pools must match
                if (avail_left[letter] - forced_left_req[letter]) != (avail_right[letter] - forced_right_req[letter]):
                    valid = False
                    break
            ans.append(valid)
            
        return ans