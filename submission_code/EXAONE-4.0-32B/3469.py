class Solution:
    def maxHeightOfTriangle(self, red: int, blue: int) -> int:
        h = 1
        max_height = 0
        while True:
            n_odd = (h + 1) // 2
            n_even = h // 2
            
            req_red1 = n_odd * n_odd
            req_blue1 = n_even * (n_even + 1)
            req_red2 = req_blue1
            req_blue2 = req_red1
            
            if (red >= req_red1 and blue >= req_blue1) or (red >= req_red2 and blue >= req_blue2):
                max_height = h
                h += 1
            else:
                break
        return max_height