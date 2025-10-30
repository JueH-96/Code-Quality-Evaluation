from typing import List

class Solution:
    def checkValidCuts(self, n: int, rectangles: List[List[int]]) -> bool:
        # The idea is that a valid horizontal (or vertical) cutting scheme
        # must partition the entire set of rectangles into three non‐empty groups
        # where every rectangle in an “earlier” group lies completely below (or left
        # of) every rectangle in a later group. In other words, if we choose a horizontal
        # cut at some y = c, then every rectangle in the bottom group must satisfy (end_y <= c)
        # and every rectangle in the top group must satisfy (start_y >= c).
        # Finally, we need two such cuts, so that the rectangles are split
        # into three groups.
        #
        # Because the only “interesting” numbers at which one may set a cut are exactly the
        # boundaries of rectangles, we will work as follows:
        #   1. To check for a potential horizontal solution, we consider the y‐coordinates.
        #      For each rectangle, we take its bottom and top (start_y and end_y) coordinates. 
        #   2. We sort the rectangles by their bottom coordinate (and then top coordinate
        #      to break ties). If we can “cut” between rectangles_0...i and rectangles_(i+1)...end
        #      meaning that the maximum top value in the first block is <= the bottom value
        #      of the next rectangle, then that position is a good candidate cut.
        #   3. If there exist two different candidate cuts (say at indices i and j, with i < j)
        #      then we have a valid horizontal partition: group1 = indices [0,i], group2 = [i+1, j],
        #      group3 = [j+1, end].
        #   4. We do a similar procedure for vertical cuts using the x‐coordinates.
        #
        # Note: The problem allows rectangles to “touch” the cut lines. That is, if a rectangle’s
        # boundary lies exactly on the cut, it is fully contained in one section. Thus we use 
        # non‐strict inequality (<=) in our comparisons.
        def can_cut(coord_idx_start: int, coord_idx_end: int) -> bool:
            # Build a list of tuples (start, end) for the coordinate we are considering.
            arr = []
            for rect in rectangles:
                arr.append((rect[coord_idx_start], rect[coord_idx_end]))
            # Sort by the start value and then by end value.
            arr.sort(key=lambda x: (x[0], x[1]))
            n_rect = len(arr)
            prefix_max = [0] * n_rect
            prefix_max[0] = arr[0][1]
            # prefix_max[i] holds the maximum "end" observed from arr[0] through arr[i].
            for i in range(1, n_rect):
                prefix_max[i] = max(prefix_max[i - 1], arr[i][1])
                
            candidate_splits = []
            # We can “cut” between i and i+1 if every rectangle from 0 to i has its end 
            # coordinate <= the start coordinate of rectangle i+1.
            for i in range(n_rect - 1):
                if prefix_max[i] <= arr[i + 1][0]:
                    candidate_splits.append(i)
            # In order to obtain three non-empty groups we need to have two such splits.
            # (The first group will be indices [0, i0], the second group [i0+1, i1],
            #  and the third group [i1+1, n_rect-1].)
            return len(candidate_splits) >= 2

        # For horizontal cuts, we use the y coordinates: bottom at index 1 and top at index 3.
        horizontal_possible = can_cut(1, 3)
        # For vertical cuts, we use the x coordinates: left at index 0 and right at index 2.
        vertical_possible = can_cut(0, 2)
        
        return horizontal_possible or vertical_possible

# The following code is provided as a driver for local testing.
# It reads input from stdin in the following format:
#
#   n k
#   r0_0 r0_1 r0_2 r0_3
#   r1_0 r1_1 r1_2 r1_3
#   ...
#   r(k-1)_0 r(k-1)_1 r(k-1)_2 r(k-1)_3
#
# For example, the first sample could be represented as:
#   5 4
#   1 0 5 2
#   0 2 2 4
#   3 2 5 3
#   0 4 4 5
#
# You can remove or comment out the section below when integrating this solution in your tests.
if __name__ == '__main__':
    import sys
    data = sys.stdin.read().strip().split()
    if not data:
        sys.exit()
    it = iter(data)
    n = int(next(it))
    k = int(next(it))
    rects = []
    for _ in range(k):
        rect = [int(next(it)) for _ in range(4)]
        rects.append(rect)
    sol = Solution()
    result = sol.checkValidCuts(n, rects)
    sys.stdout.write("true" if result else "false")