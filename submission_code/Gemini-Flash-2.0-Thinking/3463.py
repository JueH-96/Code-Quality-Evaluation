class Solution:
    def numberOfAlternatingGroups(self, colors: List[int]) -> int:
        n = len(colors)
        count = 0
        for i in range(n):
            # Indices of the three contiguous tiles
            idx1 = i
            idx2 = (i + 1) % n
            idx3 = (i + 2) % n

            # Colors of the three tiles
            color1 = colors[idx1]
            color2 = colors[idx2]
            color3 = colors[idx3]

            # Check for alternating colors
            if color1 != color2 and color3 != color2:
                count += 1
        return count