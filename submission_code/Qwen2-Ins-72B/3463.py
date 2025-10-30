class Solution:
    def numberOfAlternatingGroups(self, colors: List[int]) -> int:
        count = 0
        i = 0
        while i < len(colors):
            if i == len(colors) - 1 or colors[i] == colors[i + 1]:
                i += 1
            else:
                j = i + 1
                while j < len(colors) and colors[j] != colors[i]:
                    j += 1
                if j < len(colors):
                    count += 1
                    i = j
                else:
                    break
        return count