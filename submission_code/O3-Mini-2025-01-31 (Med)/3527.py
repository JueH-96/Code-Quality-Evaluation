from typing import List
import bisect

# We'll use Fenwick trees (Binary Indexed Trees) to support fast aggregated queries
class Fenw:
    def __init__(self, n: int):
        self.n = n
        self.fw = [0]*(n+1)
    
    def update(self, i: int, delta: int) -> None:
        # i is assumed 1-indexed.
        while i <= self.n:
            self.fw[i] += delta
            i += i & -i
    
    def query(self, i: int) -> int:
        s = 0
        while i:
            s += self.fw[i]
            i -= i & -i
        return s
    
    def query_range(self, l: int, r: int) -> int:
        if r < l:
            return 0
        return self.query(r) - self.query(l-1)
        

# The idea:
# Every contiguous alternating group of tiles (taken in the circular order) that has size s
# must have the property that every adjacent pair inside that group (there are s-1 pairs)
# alternates. Let us define an auxiliary array diff of length n:
#
#   diff[i] = 1 if colors[i] != colors[(i+1)%n], else 0.
#
# Then a contiguous block (of the circular order) of s tiles is alternating if and only if
# the corresponding window (of length s-1) in diff is all 1’s.
#
# To answer “how many alternating groups of size s exist?” we want to count the number of 
# starting indices i (0 <= i < n) for which the window diff[i], diff[i+1], …, diff[i+s-2] (mod n)
# is entirely 1’s.
#
# One way to answer such queries fast is to “compress” the structure: note that the array diff consists of
# blocks (runs) of consecutive 1’s (separated by zeros). In a non–fully-alternating circle (i.e. if
# at least one entry in diff is 0) these runs do not “wrap around” (because a wrap‐around of 1’s would imply
# that the whole circle is alternating).
#
# In a run (an interval) where diff = 1, suppose the run covers indices [L,R] (inclusive),
# so its length is rLen = R-L+1. Then this block of diff corresponds to an alternating block of tiles 
# of length (rLen + 1). And the number of contiguous subsegments (starting positions inside that run)
# of length s (which require s-1 consecutive ones in diff) is
#
#    max(0, (rLen + 1) - s + 1) = max(0, rLen - s + 2)   ,  provided (rLen+1) >= s.
#
# We maintain all these runs in a sorted list (by their starting index) and also keep aggregated information
# on run lengths in two Fenw trees: one (fenw_count) holds counts (by run length) and one (fenw_sum) holds the sum 
# of run lengths. Then for a query [1, s] (with s >= 3), if we need to count all alternating segments we sum over all
# runs that have run length L >= s-1 the contributions (L - s + 2).
#
# (A special case occurs if diff is all ones – i.e. the entire circle is alternating.
# In that case every starting index gives an alternating segment of any allowed s, so the answer is n.)
#
# When a tile’s color is updated (query type [2, idx, color]), the only diff positions that might change are:
#   pos1 = (idx - 1) mod n and pos2 = idx.
# We update diff[pos1] and diff[pos2] accordingly and adjust the runs structure (merging or splitting) and the fenw trees.
#
# We now present the complete solution.
        
class Solution:
    def numberOfAlternatingGroups(self, colors: List[int], queries: List[List[int]]) -> List[int]:
        n = len(colors)
        diff = [0]*n
        total_ones = 0
        
        # Build the diff array:
        for i in range(n):
            nxt = (i+1) % n
            if colors[i] != colors[nxt]:
                diff[i] = 1
                total_ones += 1
            else:
                diff[i] = 0
        
        # Fenw trees keyed by run length.
        # Possible run lengths in diff range from 1 to n.
        fenw_count = Fenw(n)  # counts, index = run length.
        fenw_sum = Fenw(n)    # sum of run lengths.
        
        # We'll maintain a sorted list "runs" of intervals (L, R)
        # indicating that for indices L <= i <= R in diff, we have diff[i]==1.
        runs = []  # sorted by L.
        
        # Helper routines to update the runs structure and Fenw trees:
        def add_interval(interval):
            L, R = interval
            runLength = R - L + 1
            bisect.insort_left(runs, (L, R))
            fenw_count.update(runLength, 1)
            fenw_sum.update(runLength, runLength)
            
        def remove_interval(interval):
            L, R = interval
            runLength = R - L + 1
            idx = bisect.bisect_left(runs, (L, R))
            if idx < len(runs) and runs[idx] == (L, R):
                runs.pop(idx)
            fenw_count.update(runLength, -1)
            fenw_sum.update(runLength, -runLength)
        
        # Initially, if the circle is not fully alternating, we can scan diff linearly.
        if total_ones != n:
            i = 0
            while i < n:
                if diff[i] == 1:
                    L = i
                    j = i
                    while j < n and diff[j] == 1:
                        j += 1
                    R = j - 1
                    add_interval((L, R))
                    i = j
                else:
                    i += 1
        
        # Helper: returns True if the entire circle is alternating.
        def is_full_alternating():
            return total_ones == n
        
        # When diff at a position changes, update the runs structure.
        def update_diff(pos: int, new_val: int) -> None:
            nonlocal total_ones
            old_val = diff[pos]
            if old_val == new_val:
                return
            diff[pos] = new_val
            if new_val == 1:
                total_ones += 1
                # When setting diff[pos] = 1, check whether we can merge with a run to the left and/or right.
                leftRun = None
                rightRun = None
                if pos - 1 >= 0 and diff[pos-1] == 1:
                    # find run containing pos-1
                    idx = bisect.bisect_right(runs, (pos-1, n))
                    if idx:
                        cand = runs[idx-1]
                        if cand[0] <= pos-1 <= cand[1]:
                            leftRun = cand
                if pos + 1 < n and diff[pos+1] == 1:
                    idx = bisect.bisect_left(runs, (pos+1, -1))
                    if idx < len(runs):
                        cand = runs[idx]
                        if cand[0] == pos+1:
                            rightRun = cand
                if leftRun and rightRun:
                    remove_interval(leftRun)
                    remove_interval(rightRun)
                    new_interval = (leftRun[0], rightRun[1])
                    add_interval(new_interval)
                elif leftRun:
                    remove_interval(leftRun)
                    new_interval = (leftRun[0], pos)
                    add_interval(new_interval)
                elif rightRun:
                    remove_interval(rightRun)
                    new_interval = (pos, rightRun[1])
                    add_interval(new_interval)
                else:
                    add_interval((pos, pos))
            else:
                total_ones -= 1
                # We expect pos was in some interval.
                idx = bisect.bisect_right(runs, (pos, n))
                interval = None
                if idx:
                    cand = runs[idx-1]
                    if cand[0] <= pos <= cand[1]:
                        interval = cand
                if interval is None:
                    return
                remove_interval(interval)
                L, R = interval
                # Split the interval (if non-empty parts remain)
                if L <= pos-1:
                    add_interval((L, pos-1))
                if pos+1 <= R:
                    add_interval((pos+1, R))
        
        # Process queries.
        # For query type 1: [1, s]. For each run in diff (which has run length L),
        # the corresponding alternating block in the original circle has length L+1,
        # and contributes (L+1 - s + 1) = L - s + 2 alternating segments (if L+1 >= s).
        # We use our Fenw trees keyed by run length.
        #
        # If the entire circle is alternating (i.e. diff is all 1’s), then every starting position
        # gives an alternating group – note that in a circle, even the n groups of size n (which
        # are rotations of the same set) are counted separately. So the answer in that case is n.
        result = []
        for query in queries:
            if query[0] == 1:
                s = query[1]
                if is_full_alternating():
                    result.append(n)
                else:
                    # We need to sum over runs of diff with length L >= (s-1).
                    X = s - 1
                    cnt = fenw_count.query_range(X, n)
                    totalL = fenw_sum.query_range(X, n)
                    # Each run of length L contributes (L - s + 2) to the answer.
                    ans = totalL - (s - 2) * cnt
                    result.append(ans)
            else:
                # Query type 2: update tile at index with a given new color.
                idx, newColor = query[1], query[2]
                if colors[idx] == newColor:
                    continue
                colors[idx] = newColor
                # When tile idx changes, diff at positions (idx-1)%n and idx change.
                pos1 = (idx - 1) % n
                pos2 = idx
                new_val1 = 1 if colors[pos1] != colors[(pos1+1) % n] else 0
                update_diff(pos1, new_val1)
                new_val2 = 1 if colors[pos2] != colors[(pos2+1) % n] else 0
                update_diff(pos2, new_val2)
        return result

# ---------------------
# For testing:
if __name__ == '__main__':
    sol = Solution()
    # Example 1:
    # colors = [0,1,1,0,1], queries = [[2,1,0],[1,4]]
    # Explanation: after changing tile 1 to 0, the count of alternating groups of size 4 is 2.
    print(sol.numberOfAlternatingGroups([0,1,1,0,1], [[2,1,0],[1,4]]))  # Expected [2]
    
    # Example 2:
    # colors = [0,0,1,0,1,1], queries = [[1,3],[2,3,0],[1,5]]
    # Expected output: [2,0]
    print(sol.numberOfAlternatingGroups([0,0,1,0,1,1], [[1,3],[2,3,0],[1,5]]))  # Expected [2,0]
    
    # Additional test: full alternating circle
    # For colors = [0,1,0,1] (a fully alternating sequence) and query [1,3],
    # every starting index gives an alternating segment of size 3, answer = 4.
    print(sol.numberOfAlternatingGroups([0,1,0,1], [[1,3]]))  # Expected [4]