from bisect import bisect_left, bisect_right
from typing import List

class Solution:
    def maximumCoins(self, coins: List[List[int]], k: int) -> int:
        # Sort the segments by their start coordinate.
        segs = sorted(coins, key=lambda x: x[0])
        n = len(segs)
        # Build arrays for the start (L), end (R), coin value (c), and full coin count for each segment.
        L = [seg[0] for seg in segs]
        R = [seg[1] for seg in segs]
        coin_val = [seg[2] for seg in segs]
        # For a segment covering [l, r] with coin value c,
        # its full contribution if taken entirely is c*(r-l+1).
        full = [coin_val[i] * (R[i] - L[i] + 1) for i in range(n)]
        prefix = [0] * n  # prefix sum for the full contributions
        prefix[0] = full[0]
        for i in range(1, n):
            prefix[i] = prefix[i-1] + full[i]
        
        # Explanation:
        # We wish to choose a starting bag x so that the window [x, x+k-1] collects maximum coins.
        # Each coin-segment [l, r, c] contributes c * (overlap length with [x, x+k-1]), i.e.
        #   overlap = max(0, min(r, x+k-1) - max(l, x) + 1).
        #
        # Because the coin segments are disjoint, for any fixed window:
        #   • At most one segment that starts before x may partially contribute (a “left‐partial”).
        #   • At most one segment that extends past x+k-1 may partially contribute (a “right‐partial”).
        # All segments that lie completely inside [x, x+k-1] give their full coin count.
        #
        # For a given x, let win_start = x, win_end = x+k-1.
        # • Let i_full = first index with L[i] >= x (using bisect_left on L).
        # • Let j_full = first index with R[i] > win_end (using bisect_right on R).
        # Then all segments with indices from i_full to j_full-1 have R <= win_end.
        # (They are “fully inside” provided also L[i] >= x.)
        # The segment immediately before i_full (if exists and if R[i_full-1] >= x)
        # will be only partially covered on the right (or might even cover the whole window).
        # Also, if j_full < n and the segment at j_full starts (L[j_full] ≤ win_end), then it
        # partially overlaps on the right.
        
        # Function to compute coins collected by window starting at x.
        def coinsCollected(x: int) -> int:
            win_start = x
            win_end = x + k - 1
            total = 0
            
            # Find full segments: those with L >= win_start and R <= win_end.
            i_full = bisect_left(L, win_start)
            j_full = bisect_right(R, win_end)  # first index with R > win_end
            if i_full < n and j_full - 1 >= i_full:
                total += prefix[j_full - 1] - (prefix[i_full - 1] if i_full > 0 else 0)
            
            # Left partial: segment immediately preceding i_full (if any) that overlaps the window.
            if i_full - 1 >= 0:
                idx = i_full - 1
                if R[idx] >= win_start:
                    overlap = min(R[idx], win_end) - win_start + 1
                    if overlap > 0:
                        total += coin_val[idx] * overlap
            
            # Right partial: the first segment that did NOT finish entirely inside the window.
            if j_full < n and L[j_full] <= win_end:
                overlap = win_end - L[j_full] + 1
                if overlap > 0:
                    total += coin_val[j_full] * overlap
                    
            return total
        
        # The function f(x) = coinsCollected(x) is piecewise linear and changes slope only
        # when the window’s boundaries align with a segment’s boundary.
        # Candidate starting positions come from:
        #   • x = l (for every segment l)
        #   • x such that the window’s right end equals a segment’s r, i.e. x = r - k + 1.
        # Also consider x = min(L) - k + 1 (in case an optimum window starts before any segment).
        candidates = set()
        candidates.add(L[0] - k + 1)
        for l, r, _ in segs:
            candidates.add(l)
            candidates.add(r - k + 1)
        cand_list = sorted(candidates)
        
        maxCoins = 0
        for x in cand_list:
            maxCoins = max(maxCoins, coinsCollected(x))
        return maxCoins


# Example usage and simple tests:
if __name__ == "__main__":
    sol = Solution()
    # Example 1:
    # coins = [[8,10,1],[1,3,2],[5,6,4]], k = 4
    # Best choice is window [3,4,5,6]: it overlaps segment [1,3,2] by 1 coin (at bag3)
    # and fully covers segment [5,6,4], total coins = 2 + 4*2 = 10.
    print(sol.maximumCoins([[8,10,1],[1,3,2],[5,6,4]], 4))  # Expected output: 10

    # Example 2:
    # coins = [[1,10,3]], k = 2
    # Best window: [1,2] or [9,10] yields a total of 3+3 = 6 coins.
    print(sol.maximumCoins([[1,10,3]], 2))  # Expected output: 6