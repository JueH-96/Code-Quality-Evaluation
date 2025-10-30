from typing import List
import bisect

class Solution:
    def checkValidCuts(self, n: int, rectangles: List[List[int]]) -> bool:
        # We'll write a helper function "canCut" that checks possibility
        # for a given dimension (either horizontal or vertical).
        # For horizontal, we use the y–dimension: a rectangle is represented by start_y (index 1) and end_y (index 3).
        # For vertical, we use the x–dimension: a rectangle is represented by start_x (index 0) and end_x (index 2).
        #
        # The idea is:
        #  1. Identify all candidate grid lines (cut positions) from the union of start and end coordinates that lie strictly between 0 and n.
        #  2. A grid line is "clean" (i.e. a valid cut candidate) if no rectangle is “straddling” it;
        #     in other words, for a candidate c, every rectangle satisfies either coordinate_end <= c or coordinate_start >= c.
        #     Checking this is equivalent to verifying that:
        #         count(rectangles with coordinate_start < c) - count(rectangles with coordinate_end <= c) == 0.
        #     (Because a rectangle that “crosses” c would contribute +1 to this difference.)
        #  3. If we can choose two distinct valid cut lines, call them cut1 and cut2 (with cut1 < cut2) so that
        #         • the bottom (or left) section contains at least one rectangle, i.e. group1 = (# of rectangles with coordinate_end <= cut1) > 0,
        #         • the top (or right) section contains at least one rectangle, i.e. group3 = (# of rectangles with coordinate_start >= cut2) > 0,
        #         • the middle section isn’t empty, i.e. group2 = total - group1 - group3 = (count(rectangles with coordinate_start < cut2) - group1) > 0.
        #
        # Note: When a rectangle exactly touches a cut (i.e. rectangle.end == cut1 or rectangle.start == cut2)
        # it is considered to belong entirely to one side.
        
        def canCut(rectangles: List[List[int]], n: int, startIdx: int, endIdx: int) -> bool:
            total = len(rectangles)
            # Build sorted lists of the start and end coordinates.
            starts = [rect[startIdx] for rect in rectangles]
            ends   = [rect[endIdx] for rect in rectangles]
            starts.sort()
            ends.sort()
            
            # Build a sorted list of candidate cut positions from the union of starts and ends that are strictly inside (0,n)
            candidate_set = set()
            for v in starts:
                if 0 < v < n:
                    candidate_set.add(v)
            for v in ends:
                if 0 < v < n:
                    candidate_set.add(v)
            candidates = sorted(candidate_set)
            
            # Among these candidates, we want only the "clean" ones.
            # For a candidate c, a rectangle would be "cut" by c if its coordinate_start < c < coordinate_end.
            # That happens if count_start (the number of starts < c) exceeds count_end (the number of ends <= c).
            # So c is "clean" if:
            #      bisect_left(starts, c) - bisect_right(ends, c) == 0.
            valid_cuts = []  # Each element is a tuple (cut, f) where f = number of rectangles with start < cut.
            for c in candidates:
                countS = bisect.bisect_left(starts, c)   # Number of rectangles with start < c.
                countE = bisect.bisect_right(ends, c)      # Number with end <= c.
                if countS - countE == 0:
                    # f = countS, and note that because c is "clean," 
                    # every rectangle either lies completely below (or left) or completely above (or right) of c.
                    valid_cuts.append((c, countS))
            
            # We need at least 2 valid cuts.
            if len(valid_cuts) < 2:
                return False
            
            # For a chosen cut c, the group assignment is:
            #   Group1: rectangles with coordinate_end <= c.
            #   Group3: rectangles with coordinate_start >= c.
            # For a valid partition by two cuts (cut1, cut2) with cut1 < cut2:
            #   group1 = number of rectangles with coordinate_end <= cut1 = bisect_right(ends, cut1)
            #   group3 = total - (number of rectangles with coordinate_start < cut2) = total - bisect_left(starts, cut2)
            #   group2 = total - group1 - group3 = bisect_left(starts, cut2) - group1.
            # We require each group to have at least one rectangle.
            #   That is, group1 > 0, group2 > 0 (i.e. bisect_left(starts, cut2) > group1) and group3 > 0 (i.e. bisect_left(starts, cut2) < total).
            #
            # Since valid_cuts is sorted by the candidate value (and so f = bisect_left(starts, candidate)
            # is non-decreasing), we can try to pick a valid cut1 and then search for a cut2 with the properties.
            m = len(valid_cuts)
            for i in range(m - 1):
                cut1, _ = valid_cuts[i]
                group1 = bisect.bisect_right(ends, cut1)
                if group1 == 0:
                    continue
                # We now want to find a candidate cut2 (with index > i) so that:
                #   bisect_left(starts, cut2) > group1 and bisect_left(starts, cut2) < total.
                # Because valid_cuts is sorted (and the computed f = bisect_left(starts, cut) is non-decreasing),
                # we can binary search in the valid_cuts (for indices > i) for the first candidate having f > group1.
                lo = i + 1
                hi = m - 1
                pos = -1
                while lo <= hi:
                    mid = (lo + hi) // 2
                    # valid_cuts[mid][1] is bisect_left(starts, cut2)
                    if valid_cuts[mid][1] > group1:
                        pos = mid
                        hi = mid - 1
                    else:
                        lo = mid + 1
                if pos == -1:
                    continue
                cut2, f_cut2 = valid_cuts[pos]
                # Check that group3 > 0 (i.e. f_cut2 < total) and group2 > 0 (i.e. f_cut2 - group1 > 0).
                if f_cut2 < total and (f_cut2 - group1) > 0:
                    return True
            return False
        
        # Check horizontal possibility (using y-dimension: indices 1 for start, 3 for end)
        horizontal_possible = canCut(rectangles, n, 1, 3)
        # Check vertical possibility (using x-dimension: indices 0 for start, 2 for end)
        vertical_possible = canCut(rectangles, n, 0, 2)
        
        return horizontal_possible or vertical_possible