class Solution:
    def numberOfAlternatingGroups(self, colors: List[int], queries: List[List[int]]) -> List[int]:
        n = len(colors)
        ans = []
        for query in queries:
            if query[0] == 1:
                size = query[1]
                count = 0
                for i in range(n):
                    alternating = True
                    for j in range(size - 1):
                        if colors[(i + j) % n] == colors[(i + j + 1) % n]:
                            alternating = False
                            break
                    if alternating:
                        count += 1
                ans.append(count)
            else:
                index = query[1]
                color = query[2]
                colors[index] = color
        return ans