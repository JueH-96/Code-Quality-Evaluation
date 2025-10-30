from typing import List

class Solution:
    def maxPathLength(self, coordinates: List[List[int]], k: int) -> int:
        n = len(coordinates)
        if n == 1:
            return 1

        def get_distance(p1, p2):
            return ((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2) ** 0.5

        def dfs(idx, path):
            nonlocal max_len
            if len(path) > max_len:
                max_len = len(path)

            for i in range(idx + 1, n):
                if not (path and path[-1][0] < coordinates[i][0] and path[-1][1] < coordinates[i][1]):
                    continue
                if len(path) == 1 or get_distance(path[-1], path[-2]) > get_distance(path[-1], coordinates[i]):
                    dfs(i, path + [coordinates[i]])

        def dfs2(idx, path):
            nonlocal max_len
            if len(path) > max_len:
                max_len = len(path)

            for i in range(idx + 1, n):
                if not (path and path[-1][0] < coordinates[i][0] and path[-1][1] < coordinates[i][1]):
                    continue
                if len(path) == 1 or get_distance(path[-1], path[-2]) > get_distance(path[-1], coordinates[i]):
                    dfs2(i, path + [coordinates[i]])

        max_len = 0
        for i in range(n):
            dfs(i, [coordinates[i]])
            dfs2(i, [coordinates[i]])
        return max_len