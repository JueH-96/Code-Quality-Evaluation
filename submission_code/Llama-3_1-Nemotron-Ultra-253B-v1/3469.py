class Solution:
    def maxHeightOfTriangle(self, red: int, blue: int) -> int:
        sum_total = red + blue
        h_max = 0
        while h_max * (h_max + 1) // 2 <= sum_total:
            h_max += 1
        h_max -= 1
        
        max_red = 0
        for h in range(1, h_max + 1):
            sum_r = ((h + 1) // 2) ** 2
            sum_b = (h // 2) * (h // 2 + 1)
            if sum_r <= red and sum_b <= blue:
                max_red = h
            else:
                break
        
        max_blue = 0
        for h in range(1, h_max + 1):
            sum_b = ((h + 1) // 2) ** 2
            sum_r = (h // 2) * (h // 2 + 1)
            if sum_b <= blue and sum_r <= red:
                max_blue = h
            else:
                break
        
        return max(max_red, max_blue)