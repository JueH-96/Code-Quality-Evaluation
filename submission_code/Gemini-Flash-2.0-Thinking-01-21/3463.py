class Solution:
    def numberOfAlternatingGroups(self, colors: List[int]) -> int:
        n = len(colors)
        if n < 3:
            return 0
        count = 0
        for i in range(n):
            left_index = (i - 1 + n) % n
            right_index = (i + 1) % n
            if colors[left_index] != colors[i] and colors[i] != colors[right_index]:
                count += 1
        return count