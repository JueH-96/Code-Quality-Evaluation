class Solution:
    def checkValidCuts(self, n: int, rectangles: List[List[int]]) -> bool:
        """
        We want to determine if we can partition the given set of non-overlapping rectangles
        (on an n x n grid) into three non-empty horizontal "bands" OR three non-empty vertical
        "strips" by making exactly two parallel cuts.

        A valid pair of cuts must ensure:
         1) Each resulting section contains at least one rectangle.
         2) No rectangle is "cut" by a boundary line (i.e., each rectangle lies entirely in
            exactly one of the three sections).

        -------------------------------------------------------------------------
        IDEA:

        We will check feasibility in the x-dimension and (if needed) in the y-dimension.
        If either is possible, return True; otherwise, False.

        CHECKING IN ONE DIMENSION (say the x-dimension):

        Let each rectangle i span [L[i], R[i]] along x.  A valid set of two vertical cuts
        x = x1 < x = x2 must place each rectangle i entirely in one of:
            - Left group:  R[i] <= x1
            - Middle group: L[i] >= x1  and  R[i] <= x2
            - Right group:  L[i] >= x2
        and each group must be non-empty.

        One efficient way to check:

        1) Sort the rectangles by their left edge L ascending.  (Ties can be broken arbitrarily.)
           Let m = len(rectangles).
        2) Build prefixMaxR[i] = maximum R over the first i rectangles in this sorted list
           (i.e. from index 0 to i).
        3) Build suffixMinL[i] = minimum L over the rectangles from index i to the end (i to m-1).
        4) Define a boolean array cutPossible[i] meaning:
               cutPossible[i] = True  if  prefixMaxR[i] <= suffixMinL[i+1]
           i.e. we can place a vertical cut between rectangles i and i+1.  That cut means
           "all rectangles up to i lie entirely left of the cut, and all rectangles from i+1
            onward lie entirely right of the cut," without slicing any rectangle.
        5) We need TWO such cuts, which split the set into three non-empty parts:
              group1 = [0..i], group2 = [i+1..j], group3 = [j+1..m-1].
           Concretely, we only need to find distinct indices  i < j  such that
               cutPossible[i] = True  AND  cutPossible[j] = True  AND
               i >= 0,  j <= m-2,  and  j >= i+1
           ensuring all three groups have at least one rectangle.

        If we can find two valid cut positions in x-dimension, answer is True;  otherwise we repeat
        the same logic for y-dimension.  If neither dimension yields a valid partition, return False.

        This approach is O(m log m) due to sorting, and O(m) for constructing prefix/suffix arrays
        and scanning for two valid cuts.

        -------------------------------------------------------------------------
        We'll implement a helper function check_two_cuts(rects, use_x=True) that:
        - Extracts (L,R) from rects either by x or y dimension.
        - Sorts by L, computes prefixMaxR, suffixMinL, then checks for two valid cut indices.
        - Returns True/False accordingly.

        Finally, we return check_two_cuts(...) or check_two_cuts(..., use_x=False).
        """

        import sys
        sys.setrecursionlimit(10**7)

        def check_two_cuts(rects, use_x=True):
            """
            rects: list of [x1, y1, x2, y2]
            use_x=True  => consider the x-intervals (x1, x2)
            use_x=False => consider the y-intervals (y1, y2)

            Returns True if we can find two parallel cuts that partition rects
            into three non-empty sections without slicing any rectangle.
            """
            m = len(rects)
            if m < 3:
                return False  # need at least 3 rectangles to form 3 groups

            # Extract intervals: L[i], R[i]
            if use_x:
                intervals = [(r[0], r[2]) for r in rects]  # (left, right)
            else:
                intervals = [(r[1], r[3]) for r in rects]  # (bottom, top)

            # Sort by L ascending
            intervals.sort(key=lambda x: x[0])
            # Build prefixMaxR[i] = max R among intervals[0..i]
            prefixMaxR = [0]*m
            prefixMaxR[0] = intervals[0][1]
            for i in range(1, m):
                prefixMaxR[i] = max(prefixMaxR[i-1], intervals[i][1])
            # Build suffixMinL[i] = min L among intervals[i..m-1]
            suffixMinL = [0]*m
            suffixMinL[m-1] = intervals[m-1][0]
            for i in range(m-2, -1, -1):
                suffixMinL[i] = min(suffixMinL[i+1], intervals[i][0])

            # cutPossible[i] = True if we can place a cut between i and i+1
            # i.e. prefixMaxR[i] <= suffixMinL[i+1]
            cutPossible = [False]*(m-1)
            for i in range(m-1):
                if prefixMaxR[i] <= suffixMinL[i+1]:
                    cutPossible[i] = True

            # We need to find two distinct indices i<j such that cutPossible[i] and cutPossible[j] are True,
            # with all groups non-empty => i >= 0, j <= m-2, j >= i+1
            # group1 = [0..i], group2 = [i+1..j], group3 = [j+1..m-1].
            valid_cuts = [idx for idx, can_cut in enumerate(cutPossible) if can_cut]

            # We need at least 2 valid cut positions. Let them be i < j.
            # Also j must be <= m-2 to leave room for group3 (j+1..end non-empty).
            # And i >= 0 is automatically true if i is in range(m-1).
            # We just check if among valid_cuts, there exist two distinct indices
            # i < j <= m-2. The second condition "j <= m-2" is needed to ensure group3 isn't empty.
            # The middle group is [i+1..j], so we also need i+1 <= j => j >= i+1 => i < j anyway.

            # A simple way: scan valid_cuts. For each i in valid_cuts, we look for a j in valid_cuts
            # with j>i. Also check j <= m-2.
            # If found any, return True.

            # Because valid_cuts is sorted, we just need to see if there's at least two distinct
            # valid cut positions and that the largest one is <= m-2.
            if len(valid_cuts) < 2:
                return False

            # If the second-to-last or last valid cut is <= m-2, we can form two cuts
            # as long as we have two distinct ones.
            # More precisely: if the largest valid cut is <= m-2, then we just need
            # at least 2 distinct elements in valid_cuts. The smallest is i, the next is j.
            # i < j <= that largest.
            # So we check if valid_cuts[0] < valid_cuts[-1]  and valid_cuts[-1] <= m-2.
            if valid_cuts[-1] <= m-2 and valid_cuts[0] < valid_cuts[-1]:
                return True

            return False

        m = len(rectangles)

        # Check vertical possibility
        if check_two_cuts(rectangles, use_x=True):
            return True
        # Check horizontal possibility
        if check_two_cuts(rectangles, use_x=False):
            return True
        return False