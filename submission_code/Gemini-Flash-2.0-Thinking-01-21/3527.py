class Solution:
    def numberOfAlternatingGroups(self, colors: List[int], queries: List[List[int]]) -> List[int]:
        def is_alternating_group(current_colors, start_index, size):
            n = len(current_colors)
            group = []
            for i in range(size):
                group.append(current_colors[(start_index + i) % n])
            for i in range(size - 1):
                if group[i] == group[i+1]:
                    return False
            return True

        def count_alternating_groups(current_colors, size):
            n = len(current_colors)
            count = 0
            for start_index in range(n):
                if is_alternating_group(current_colors, start_index, size):
                    count += 1
            return count

        current_colors = list(colors)
        results = []
        for query in queries:
            if query[0] == 1:
                size = query[1]
                results.append(count_alternating_groups(current_colors, size))
            elif query[0] == 2:
                index = query[1]
                color = query[2]
                current_colors[index] = color
        return results