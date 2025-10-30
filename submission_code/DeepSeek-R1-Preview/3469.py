class Solution:
    def maxHeightOfTriangle(self, red: int, blue: int) -> int:
        total = red + blue
        h_max = 0
        while h_max * (h_max + 1) // 2 <= total:
            h_max += 1
        h_max -= 1
        
        def compute_max(start_red: bool) -> int:
            for h in range(h_max, 0, -1):
                if start_red:
                    sum_r = ((h + 1) // 2) ** 2
                    sum_b = (h // 2) * (h // 2 + 1)
                else:
                    sum_b = ((h + 1) // 2) ** 2
                    sum_r = (h // 2) * (h // 2 + 1)
                if sum_r <= red and sum_b <= blue:
                    return h
            return 0
        
        h_red = compute_max(True)
        h_blue = compute_max(False)
        return max(h_red, h_blue)