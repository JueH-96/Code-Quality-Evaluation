class Solution:
    def numberOfAlternatingGroups(self, colors: List[int]) -> int:
        n = len(colors)
        count = 0
        for i in range(n):
            left = colors[(i - 1 + n) % n]
            mid = colors[i]
            right = colors[(i + 1) % n]
            if left != mid and mid != right:
                count += 1
        return count