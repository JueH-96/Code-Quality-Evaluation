class Solution:
    def numberOfAlternatingGroups(self, colors: List[int]) -> int:
        n = len(colors)
        count = 0

        # Extend the list to handle the circular nature
        extended_colors = colors + colors[:2]

        # Check for alternating groups
        for i in range(n):
            if extended_colors[i] != extended_colors[i + 1] and extended_colors[i + 1] != extended_colors[i + 2] and extended_colors[i] != extended_colors[i + 2]:
                count += 1

        return count