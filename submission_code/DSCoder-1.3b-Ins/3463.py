class Solution:
    def numberOfAlternatingGroups(self, colors: List[int]) -> int:
        count = 1
        i = 0
        while i < len(colors) - 2:
            if colors[i] == colors[i+1]:
                i += 2
            else:
                while i < len(colors) - 1 and colors[i] == colors[i+1]:
                    i += 1
                count += 1
                i += 1
        return count