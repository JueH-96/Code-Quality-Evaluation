class Solution:
    def maxRectangleArea(self, points: List[List[int]]) -> int:
        points.sort()
        p = defaultdict(list)
        for ix, (x, y) in enumerate(points):
            p[y].append(ix)
        n = len(points)
        next_greater = [n] * n
        stack = []
        for i in range(n):
            while stack and points[stack[-1]][1] < points[i][1]:
                j = stack.pop()
                next_greater[j] = i
            stack.append(i)
        
        prev_smaller, next_greater = [-1] * n, [n] * n
        stack = []
        for i in range(n):
            while stack and points[stack[-1]][1] >= points[i][1]:
                j = stack.pop()
                prev_smaller[j] = i
                next_greater[i] = min(next_greater[i], next_greater[j])
            stack.append(i)
        
        res = -1
        for y, I in p.items():
            I = [-1] + I + [n]
            for i in range(1, len(I) - 1):
                res = max(res, (I[i+1] - I[i-1]) * (points[I[i]][1] - y))
        return res