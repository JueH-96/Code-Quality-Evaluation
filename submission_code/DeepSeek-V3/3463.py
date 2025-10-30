class Solution:
    def numberOfAlternatingGroups(self, colors: List[int]) -> int:
        n = len(colors)
        count = 0
        for i in range(n):
            # Get the three tiles: current, next, and next next
            # Since it's a circle, use modulo to wrap around
            a = colors[i]
            b = colors[(i + 1) % n]
            c = colors[(i + 2) % n]
            # Check if the middle tile is different from both left and right
            if b != a and b != c:
                count += 1
        return count