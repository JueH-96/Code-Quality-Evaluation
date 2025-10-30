from typing import List
from collections import Counter

class Solution:
    def checkValidCuts(self, n: int, rectangles: List[List[int]]) -> bool:
        def check_horizontal(rectangles):
            tops = Counter()
            bottoms = Counter()
            top_min, top_max = n, 0
            bottom_min, bottom_max = n, 0

            for rect in rectangles:
                bottom, top = rect[1], rect[3]
                tops[top] += 1
                bottoms[bottom] += 1
                top_min = min(top_min, top)
                top_max = max(top_max, top)
                bottom_min = min(bottom_min, bottom)
                bottom_max = max(bottom_max, bottom)

            # Early pruning possibility
            if top_min > bottom_max + 1:
                return False

            for pos, count in tops.items():
                if count > len(rectangles) // 3:
                    return False

            for pos, count in bottoms.items():
                if count > len(rectangles) // 3:
                    return False

            # Try cuts
            for cut1 in range(bottom_max + 1, top_min):
                for cut2 in range(cut1 + 1, top_max + 1):
                    above_cut = len(rectangles) - len(tops) - len(bottoms)
                    for pos, count in tops.items():
                        if pos > cut2:
                            above_cut += count
                    for pos, count in bottoms.items():
                        if pos < cut1:
                            above_cut += count
                    if above_cut >= len(rectangles) // 3:
                        below_cut = len-bottoms
                        for pos, count in bottoms.items():
                            if pos < cut1:
                                below_cut += count
                        for pos, count in tops.items():
                            if pos < cut1 and pos > cut2:
                                below_cut += count
                        if below_cut >= len(rectangles) // 3:
                            return True

            return False

        def check_vertical(rectangles):
            return check_horizontal([[rect[1], rect[0], rect[3], rect[2]] for rect in rectangles])

        if len(rectangles) < 3:
            return False

        return check_horizontal(rectangles) or check_vertical(rectangles)