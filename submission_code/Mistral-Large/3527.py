from typing import List

class Solution:
    def numberOfAlternatingGroups(self, colors: List[int], queries: List[List[int]]) -> List[int]:
        n = len(colors)
        result = []

        def count_alternating_groups(size):
            count = 0
            for i in range(n):
                group_size = 1
                for j in range(1, size):
                    if colors[(i + j) % n] != colors[(i + j - 1) % n]:
                        group_size += 1
                    else:
                        break
                if group_size == size:
                    count += 1
            return count

        for query in queries:
            if query[0] == 1:
                size = query[1]
                result.append(count_alternating_groups(size))
            elif query[0] == 2:
                index, color = query[1], query[2]
                colors[index] = color

        return result