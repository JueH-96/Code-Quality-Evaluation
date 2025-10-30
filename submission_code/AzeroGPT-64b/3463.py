class Solution:
    def numberOfAlternatingGroups(self, colors: List[int]) -> int:
        cnt, is_red, n = 0, colors[0], len(colors)
        for color in colors:
            need = 1 - is_red if cnt > 0 else colors[0]
            if color == need:
                cnt += 1
                is_red = 1 - is_red
        return cnt + (colors[0] == colors[-1])