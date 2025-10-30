class Solution:
    def numberOfAlternatingGroups(self, colors: List[int], queries: List[List[int]]) -> List[int]:
        answers = []
        n = len(colors)

        for query in queries:
            if query[0] == 1:
                size = query[1]
                count = 0
                for i in range(n):
                    is_alternating = True
                    for j in range(size - 1):
                        idx1 = (i + j) % n
                        idx2 = (i + j + 1) % n
                        if colors[idx1] == colors[idx2]:
                            is_alternating = False
                            break
                    if is_alternating:
                        count += 1
                answers.append(count)
            elif query[0] == 2:
                index = query[1]
                color = query[2]
                colors[index] = color

        return answers