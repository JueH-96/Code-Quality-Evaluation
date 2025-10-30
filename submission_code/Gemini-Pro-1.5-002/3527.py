class Solution:
    def numberOfAlternatingGroups(self, colors: List[int], queries: List[List[int]]) -> List[int]:
        n = len(colors)
        ans = []
        for q in queries:
            if q[0] == 1:
                size = q[1]
                count = 0
                for i in range(n):
                    alt = True
                    for j in range(1, size):
                        if colors[(i + j) % n] == colors[(i + j - 1) % n]:
                            alt = False
                            break
                    if alt:
                        count += 1
                ans.append(count)
            else:
                index = q[1]
                color = q[2]
                colors[index] = color
        return ans