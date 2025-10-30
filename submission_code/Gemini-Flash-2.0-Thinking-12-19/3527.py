class Solution:
    def numberOfAlternatingGroups(self, colors: List[int], queries: List[List[int]]) -> List[int]:
        def is_alternating(group):
            if len(group) < 2:
                return True
            for i in range(1, len(group)):
                if group[i] == group[i-1]:
                    return False
            return True

        def get_circular_group(start_index, size, colors):
            n = len(colors)
            group = []
            for i in range(size):
                index = (start_index + i) % n
                group.append(colors[index])
            return group

        def count_alternating_groups(size, colors):
            count = 0
            n = len(colors)
            if size > n:
                return 0
            for start_index in range(n):
                group = get_circular_group(start_index, size, colors)
                if is_alternating(group):
                    count += 1
            return count

        answer = []
        current_colors = list(colors)
        for query in queries:
            if query[0] == 1:
                size_i = query[1]
                answer.append(count_alternating_groups(size_i, current_colors))
            elif query[0] == 2:
                index_i = query[1]
                color_i = query[2]
                current_colors[index_i] = color_i
        return answer