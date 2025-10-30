# YOUR CODE HERE
import sys
from collections import defaultdict

def solve():
    input = sys.stdin.read
    def solve():
        N, D = map(int, input().split())
        points = [list(map(int, input().split())) for _ in range(N)]
        x_points = sorted([p[0] for p in points])
        y_points = sorted([p[1] for p in points])
        x_count = defaultdict(int)
        y_count = defaultdict(int)
        for p in points:
            x_count[p[0]] += 1
            y_count[p[1]] += 1
        x_prefix = [0]
        y_prefix = [0]
        for i in range(len(x_points)):
            x_prefix.append(x_prefix[-1] + x_count[x_points[i]])
        for i in range(len(y_points)):
            y_prefix.append(y_prefix[-1] + y_count[y_points[i]])
        ans = 0
        for i in range(len(x_points)):
            for j in range(len(y_points)):
                left = x_points[i] - D
                right = x_points[i] + D
                bottom = y_points[j] - D
                top = y_points[j] + D
                if left < x_points[0]:
                    left = x_points[0]
                if right > x_points[-1]:
                    right = x_points[-1]
                if bottom < y_points[0]:
                    bottom = y_points[0]
                if top > y_points[-1]:
                    top = y_points[-1]
                if left > right or bottom > top:
                    continue
                x_left = x_prefix[bisect_left(x_points, left)]
                x_right = x_prefix[bisect_right(x_points, right)]
                y_bottom = y_prefix[bisect_left(y_points, bottom)]
                y_top = y_prefix[bisect_right(y_points, top)]
                ans += (x_right - x_left) * (y_top - y_bottom)
        return ans

    print(solve())

from bisect import bisect_left, bisect_right
solve()