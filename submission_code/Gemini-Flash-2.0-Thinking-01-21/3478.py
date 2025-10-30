class Solution:
    def canReachCorner(self, xCorner: int, yCorner: int, circles: List[List[int]]) -> bool:
        n_circles = len(circles)
        parent = list(range(n_circles))

        def find_set(i):
            if parent[i] == i:
                return i
            parent[i] = find_set(parent[i])
            return parent[i]

        def union_sets(i, j):
            i_id = find_set(i)
            j_id = find_set(j)
            if i_id != j_id:
                parent[i_id] = j_id

        for i in range(n_circles):
            for j in range(i + 1, n_circles):
                x1, y1, r1 = circles[i]
                x2, y2, r2 = circles[j]
                dist_sq = (x1 - x2)**2 + (y1 - y2)**2
                if dist_sq <= (r1 + r2)**2:
                    union_sets(i, j)

        component_flags = {}
        for i in range(n_circles):
            root = find_set(i)
            if root not in component_flags:
                component_flags[root] = {'left': False, 'right': False, 'bottom': False, 'top': False}
            x_c, y_c, r_c = circles[i]
            if x_c - r_c <= 0:
                component_flags[root]['left'] = True
            if x_c + r_c >= xCorner:
                component_flags[root]['right'] = True
            if y_c - r_c <= 0:
                component_flags[root]['bottom'] = True
            if y_c + r_c >= yCorner:
                component_flags[root]['top'] = True

        for root in component_flags:
            flags = component_flags[root]
            if (flags['left'] and flags['right']) or (flags['bottom'] and flags['top']):
                return False

        return True