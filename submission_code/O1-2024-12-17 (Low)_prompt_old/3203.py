class Solution:
    def canMakePalindromeQueries(self, s: str, queries: List[List[int]]) -> List[bool]:
        import sys
        input_data = sys.stdin

        # ------------------------------------------
        # 1) First, check overall feasibility by comparing
        #    the total frequency of each character in the left half
        #    vs. the right half. If they differ for any character,
        #    then no query can ever make the string a palindrome.
        # ------------------------------------------
        n = len(s)
        half = n // 2
        from collections import Counter
        
        left_half_count = Counter(s[:half])
        right_half_count = Counter(s[half:])
        if left_half_count != right_half_count:
            # Frequencies differ, so all queries answer False
            return [False]*len(queries)
        
        # ------------------------------------------
        # 2) Build an array mismatchMask[] of length n//2
        #    mismatchMask[i] = 1 if s[i] != s[n-1-i], else 0.
        #
        #    Then build a prefix sum array mismatchPS of length n//2+1
        #    where mismatchPS[k] = sum of mismatchMask[0..k-1].
        #    So mismatchPS[0] = 0, mismatchPS[i+1] = mismatchPS[i] + mismatchMask[i].
        #
        #    totalMismatch = total number of mismatched pairs.
        # ------------------------------------------
        mismatchMask = [0]*(half)
        for i in range(half):
            if s[i] != s[n-1-i]:
                mismatchMask[i] = 1
        
        mismatchPS = [0]*(half+1)
        for i in range(half):
            mismatchPS[i+1] = mismatchPS[i] + mismatchMask[i]
        totalMismatch = mismatchPS[half]
        
        # ------------------------------------------
        # 3) For each query [a,b,c,d]:
        #    - We can reorder indices in left half [a..b].
        #    - We can reorder indices in right half [c..d].
        #    - But the right-half indices [c..d] correspond to
        #      left-half indices [n-1-d .. n-1-c] when looking
        #      at mismatchMask's index i (which pairs with n-1-i).
        #
        #    Let leftRange = [a, b].
        #    Let rightRange = [R1, R2] = [n-1-d, n-1-c],
        #      both ranges in [0..half-1].
        #
        #    A mismatch at index i is fixable if i is in leftRange
        #    or i is in rightRange. If i is not in either range
        #    and mismatchMask[i] = 1, we cannot fix it => answer = False.
        #
        #    So we just need to check if all mismatched indices i
        #    lie in (leftRange U rightRange).
        #
        #    We do this by:
        #      sumUnion = (# of mismatch in leftRange) + (# of mismatch in rightRange)
        #                  - (# of mismatch in intersection).
        #      If sumUnion == totalMismatch => we can fix all => True
        #      else => False
        #
        #    We can get # of mismatch in [L..R] using prefix sums:
        #       mismatchPS[R+1] - mismatchPS[L].
        # ------------------------------------------
        
        ans = []
        
        for (a, b, c, d) in queries:
            # Left range:
            L1, L2 = a, b
            
            # Right range in terms of left-half indices:
            R1, R2 = (n-1-d), (n-1-c)
            # They should be within [0, half-1] as per constraints.
            
            # Sum of mismatches in leftRange:
            sumLeft = mismatchPS[L2+1] - mismatchPS[L1]
            
            # Sum of mismatches in rightRange:
            sumRight = mismatchPS[R2+1] - mismatchPS[R1]
            
            # Intersection range:
            interL = max(L1, R1)
            interR = min(L2, R2)
            if interR >= interL:
                sumInter = mismatchPS[interR+1] - mismatchPS[interL]
            else:
                sumInter = 0
            
            sumUnion = sumLeft + sumRight - sumInter
            if sumUnion == totalMismatch:
                ans.append(True)
            else:
                ans.append(False)
        
        return ans