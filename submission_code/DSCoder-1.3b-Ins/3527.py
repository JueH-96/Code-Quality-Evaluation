class Solution:
    def numberOfAlternatingGroups(self, colors: List[int], queries: List[List[int]]) -> List[int]:
        def count_groups(colors, size):
            groups = 0
            for i in range(size):
                if i == 0 or i == size - 1:
                    groups += 1
                elif colors[i] != colors[i - 1]:
                    groups += 1
            return groups

        def change_color(colors, index, color):
            colors[index] = color
            return colors

        result = []
        for query in queries:
            if query[0] == 1:
                result.append(count_groups(colors, query[1]))
            elif query[0] == 2:
                colors = change_color(colors, query[1], query[2])
        return result