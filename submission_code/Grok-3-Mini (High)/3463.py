class Solution:
    def numberOfAlternatingGroups(self, colors: List[int]) -> int:
        n = len(colors)
        count = 0
        for m in range(n):
            left_idx = (m - 1) % n
            right_idx = (m + 1) % n
            if colors[m] != colors[left_idx] and colors[m] != colors[right_idx]:
                count += 1
        return count