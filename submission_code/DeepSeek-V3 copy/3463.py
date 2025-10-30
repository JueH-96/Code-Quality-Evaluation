class Solution:
    def numberOfAlternatingGroups(self, colors: List[int]) -> int:
        n = len(colors)
        count = 0
        for i in range(n):
            # Get the indices of the three tiles
            a = colors[i]
            b = colors[(i+1)%n]
            c = colors[(i+2)%n]
            # Check if they form an alternating group
            if a != b and b != c:
                count += 1
        return count