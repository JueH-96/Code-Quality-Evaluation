class Solution:
    def numberOfAlternatingGroups(self, colors: List[int]) -> int:
        count = 0
        for i in range(len(colors) - 2):
            if colors[i] != colors[i+2]:
                count += 1
        return count