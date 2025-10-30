class Solution:
    def numberOfAlternatingGroups(self, colors: List[int]) -> int:
        count = 0
        n = len(colors)
        for i in range(n):
            prev = (i - 1) % n
            next = (i + 1) % n
            if colors[i] != colors[prev] and colors[i] != colors[next]:
                count += 1
        return count