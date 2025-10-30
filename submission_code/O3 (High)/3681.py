from typing import List

class Solution:
    def maxRectangleArea(self, points: List[List[int]]) -> int:
        """
        Brute-force all axis-aligned rectangles that can be formed with the
        points as corners.  A rectangle is valid if:
          • the four corner points exist in the input
          • no other point lies inside it or on its border
        The input size is ≤ 10, so an O(n³) enumeration is more than fast enough.
        """
        # Put every point in a set for O(1) existence checks
        pts = set(map(tuple, points))

        # Distinct x and y coordinates present in the input
        xs = sorted({p[0] for p in points})
        ys = sorted({p[1] for p in points})

        best = -1  # maximum area found so far

        # Choose two different x’s and two different y’s to define a rectangle
        for i in range(len(xs)):
            for j in range(i + 1, len(xs)):
                x1, x2 = xs[i], xs[j]
                for k in range(len(ys)):
                    for l in range(k + 1, len(ys)):
                        y1, y2 = ys[k], ys[l]

                        # The four required corner points
                        corners = [(x1, y1), (x1, y2), (x2, y1), (x2, y2)]
                        if not all(c in pts for c in corners):
                            continue  # at least one corner is missing

                        # Verify that no OTHER point is on the border or inside
                        valid = True
                        for px, py in pts:
                            if (px, py) in corners:
                                continue
                            if x1 <= px <= x2 and y1 <= py <= y2:
                                valid = False
                                break

                        if valid:
                            area = (x2 - x1) * (y2 - y1)
                            best = max(best, area)

        return best