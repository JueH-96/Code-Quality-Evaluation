class Solution:
    def numberOfAlternatingGroups(self, colors: List[int], queries: List[List[int]]) -> List[int]:
        n = len(colors)
        prefix = [[0, 0] for _ in range(n+1)]
        for i in range(n):
            prefix[i+1][0] = prefix[i][0] + (colors[i] == 0)
            prefix[i+1][1] = prefix[i][1] + (colors[i] == 1)
        
        ans = []
        for q in queries:
            if q[0] == 1:
                size = q[1]
                count = 0
                for i in range(size, n+1):
                    count += (prefix[i][1] - prefix[i-size][1]) == (prefix[i][0] - prefix[i-size][0]) == size // 2
                ans.append(count)
            else:
                index = q[1]
                color = q[2]
                if colors[index] != color:
                    colors[index] = color
                    for i in range(index+1, n+1):
                        prefix[i][0] = prefix[i-1][0] + (colors[i-1] == 0)
                        prefix[i][1] = prefix[i-1][1] + (colors[i-1] == 1)
                    for i in range(index, -1, -1):
                        prefix[i][0] = prefix[i+1][0] - (colors[i] == 0)
                        prefix[i][1] = prefix[i+1][1] - (colors[i] == 1)
        return ans