from typing import List

class Solution:
    def canMakePalindromeQueries(self, s: str, queries: List[List[int]]) -> List[bool]:
        n = len(s)
        half = n // 2
        
        # Precompute the mismatch array for i in [0, half)
        mismatch = [0] * half
        for i in range(half):
            if s[i] != s[n - 1 - i]:
                mismatch[i] = 1
        
        # Build prefix sum array for mismatch.
        # p[i] = sum(mismatch[0 ... i-1])
        p = [0] * (half + 1)
        for i in range(half):
            p[i+1] = p[i] + mismatch[i]
        
        # Helper function to get the sum of mismatch in an interval [l, r] inclusive
        def get_sum(l: int, r: int) -> int:
            if l > r:
                return 0
            return p[r+1] - p[l]
        
        result = []
        for a, b, c, d in queries:
            # I1 from first half allowed: indices [a, b] intersect [0, half-1]
            L1 = max(a, 0)
            R1 = min(b, half - 1)
            intervals = []
            if L1 <= R1:
                intervals.append((L1, R1))
                
            # I2 from second half allowed for mirror indices:
            # n-1-i allowed when i in [n-1-d, n-1-c]
            L2 = max(n - 1 - d, 0)
            R2 = min(n - 1 - c, half - 1)
            if L2 <= R2:
                intervals.append((L2, R2))
            
            # Merge intervals in intervals (they may overlap)
            if intervals:
                intervals.sort()
                merged = []
                cur_l, cur_r = intervals[0]
                for i in range(1, len(intervals)):
                    l, r = intervals[i]
                    if l <= cur_r + 1:  # overlapping or adjacent
                        cur_r = max(cur_r, r)
                    else:
                        merged.append((cur_l, cur_r))
                        cur_l, cur_r = l, r
                merged.append((cur_l, cur_r))
            else:
                merged = []
            
            # Total fixed positions in [0, half)
            total_fixed_mismatch = get_sum(0, half-1)
            # Subtract the ones in allowed intervals (free indices)
            for l, r in merged:
                total_fixed_mismatch -= get_sum(l, r)
            
            result.append(total_fixed_mismatch == 0)
        
        return result

# Example usage:
# sol = Solution()
# print(sol.canMakePalindromeQueries("abcabc", [[1,1,3,5],[0,2,5,5]]))  # Expected: [True, True]