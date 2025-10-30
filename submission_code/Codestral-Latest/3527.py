class Solution:
    def numberOfAlternatingGroups(self, colors: List[int], queries: List[List[int]]) -> List[int]:
        n = len(colors)
        colors = colors + colors  # Extend the colors array to handle circular arrangement

        def count_alternating_groups(size):
            count = 0
            for i in range(n):
                if all(colors[i + j] != colors[i + j + 1] for j in range(size - 1)):
                    count += 1
            return count

        answer = []
        for query in queries:
            if query[0] == 1:
                size = query[1]
                answer.append(count_alternating_groups(size))
            elif query[0] == 2:
                index, color = query[1], query[2]
                colors[index] = color
                colors[index + n] = color  # Update the extended array as well

        return answer