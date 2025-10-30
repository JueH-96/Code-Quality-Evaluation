class Solution:
    def numberOfAlternatingGroups(self, colors):
        colors.append(colors[0])
        colors.append(colors[1])
        count = 0
        for i in range(len(colors) - 2):
            if colors[i] != colors[i + 1] and colors[i + 1] != colors[i + 2] and colors[i] != colors[i + 2]:
                count += 1
        return count