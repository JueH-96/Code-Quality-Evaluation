class Solution:
    def numberOfAlternatingGroups(self, colors: List[int]) -> int:
        n = len(colors)
        count = 0
        for i in range(n):
            j = (i + 1) % n
            k = (i + 2) % n
            if colors[j] != colors[i] and colors[j] != colors[k]:
                count += 1
        return count