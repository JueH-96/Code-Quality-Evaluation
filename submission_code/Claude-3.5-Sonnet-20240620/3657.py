class Solution:
    def checkValidCuts(self, n: int, rectangles: List[List[int]]) -> bool:
        def can_cut(coords):
            coords.sort()
            left, mid, right = set(), set(), set()
            
            for i, (start, end) in enumerate(coords):
                if not left:
                    left.add(i)
                elif not mid and start > coords[list(left)[-1]][1]:
                    mid.add(i)
                else:
                    right.add(i)
            
            return left and mid and right and max(coords[i][1] for i in left) < min(coords[i][0] for i in mid) < max(coords[i][1] for i in mid) < min(coords[i][0] for i in right)

        # Check horizontal cuts
        horizontal_coords = [(rect[1], rect[3]) for rect in rectangles]
        if can_cut(horizontal_coords):
            return True

        # Check vertical cuts
        vertical_coords = [(rect[0], rect[2]) for rect in rectangles]
        if can_cut(vertical_coords):
            return True

        return False