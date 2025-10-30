class Solution:
    def numberOfAlternatingGroups(self, colors: List[int]) -> int:
        n = len(colors)
        count = 0
        for i in range(n):
            c1 = colors[i]
            c2 = colors[(i + 1) % n]
            c3 = colors[(i + 2) % n]
            if c1 != c2 and c2 != c3:
                count += 1
        return count