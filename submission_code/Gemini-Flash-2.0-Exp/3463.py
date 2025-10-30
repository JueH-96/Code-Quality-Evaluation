class Solution:
    def numberOfAlternatingGroups(self, colors: List[int]) -> int:
        n = len(colors)
        count = 0
        for i in range(n):
            prev = colors[i - 1]
            curr = colors[i]
            next_tile = colors[(i + 1) % n]
            if prev != curr and next_tile != curr:
                count += 1
        return count