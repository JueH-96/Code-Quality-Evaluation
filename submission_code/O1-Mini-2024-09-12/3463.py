class Solution:
    def numberOfAlternatingGroups(self, colors: List[int]) -> int:
        n = len(colors)
        count = 0
        for i in range(n):
            left = colors[i - 1]
            middle = colors[i]
            right = colors[(i + 1) % n]
            if left == right and middle != left:
                count += 1
        return count