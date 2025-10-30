from typing import List

class Solution:
    def maxSelectedElements(self, nums: List[int]) -> int:
        # We have intervals [v, v+1] for each original v = nums[i].
        # We want the longest integer-consecutive sequence [L..R]
        # such that we can pick distinct intervals to cover each integer.
        # Hall's theorem for these unit-intervals reduces to:
        #   for a window [L..R] of length k, the number of intervals
        #   that intersect [L..R] must be at least k.
        # An interval [v, v+1] intersects [L..R] iff v <= R and v+1 >= L
        #   i.e. v in [L-1..R].
        # Let freq[x] = count of nums equal to x.
        # Let pre[x] = prefix sum of freq up to x.
        # Then the number of intervals intersecting [L..R] is
        #   pre[R] - pre[L-2]   (i.e., count of v in [L-1..R]).
        # We slide L and extend R as far as possible so that
        #   (R - L + 1) <= number of intersecting intervals.
        # Complexity O(M), M = max(nums), which is <= 1e6.
        
        if not nums:
            return 0
        
        maxv = max(nums)
        M = maxv + 1  # we need up to index maxv+1 in freq
        
        # Build frequency array
        freq = [0] * (M + 1)
        for v in nums:
            freq[v] += 1
        
        # Build prefix sums pre[i] = sum freq[0..i]
        pre = [0] * (M + 1)
        running = 0
        for i in range(M + 1):
            running += freq[i]
            pre[i] = running
        
        ans = 0
        R = 0
        # Slide L from 1 to M (possible sequence starts)
        # Note: L could be as small as 1 (we can't cover 0 unless some num= -1,
        # but nums>=1 so minimal coverage starts at 1).
        for L in range(1, M + 1):
            # Ensure R >= L-1 so window is valid
            if R < L - 1:
                R = L - 1
            # Try to extend R while feasible
            # window length k = R-L+1, intervals = pre[R] - pre[L-2]
            # We test if length+1 <= intervals when R moves to R+1
            while R + 1 <= M:
                k = (R + 1) - L + 1
                # count of v in [L-1 .. R+1] = pre[R+1] - pre[L-2]
                low = L - 2
                cnt = pre[R + 1] - (pre[low] if low >= 0 else 0)
                if cnt >= k:
                    R += 1
                else:
                    break
            # update answer
            ans = max(ans, R - L + 1)
        
        return ans