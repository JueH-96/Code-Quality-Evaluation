class Solution:
    def numberOfAlternatingGroups(self, colors: List[int], queries: List[List[int]]) -> List[int]:
        def count_alternating_groups(size):
            count = 0
            for i in range(len(colors)):
                if all(colors[(i + j) % len(colors)] != colors[(i + j + 1) % len(colors)] for j in range(size - 1)):
                    count += 1
            return count // size

        answer = []
        for query in queries:
            if query[0] == 1:
                size = query[1]
                answer.append(count_alternating_groups(size))
            else:
                index, color = query[1], query[2]
                colors[index] = color
        return answer