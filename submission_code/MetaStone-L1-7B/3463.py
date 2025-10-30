class Solution:
    def numberOfAlternatingGroups(self, colors: List[int]) -> int:
        n = len(colors)
        count = 0
        for i in range(n):
            a = colors[i]
            b = colors[(i + 1) % n]
            c = colors[(i + 2) % n]
            if b != a and b != c:
                count += 1
        return count