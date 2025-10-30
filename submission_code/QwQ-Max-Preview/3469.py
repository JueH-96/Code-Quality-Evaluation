class Solution:
    def maxHeightOfTriangle(self, red: int, blue: int) -> int:
        total = red + blue
        max_h = self.find_max_h(total)
        for h in range(max_h, 0, -1):
            if self.check_configuration(h, red, blue, start_red=True):
                return h
            if self.check_configuration(h, red, blue, start_red=False):
                return h
        return 0
    
    def find_max_h(self, total):
        low, high = 1, total
        best = 0
        while low <= high:
            mid = (low + high) // 2
            current_sum = mid * (mid + 1) // 2
            if current_sum <= total:
                best = mid
                low = mid + 1
            else:
                high = mid - 1
        return best
    
    def check_configuration(self, h, red, blue, start_red):
        if start_red:
            sum_red, sum_blue = self.calculate_sums(h, start_red=True)
        else:
            sum_blue, sum_red = self.calculate_sums(h, start_red=False)
        return red >= sum_red and blue >= sum_blue
    
    def calculate_sums(self, h, start_red):
        if start_red:
            if h % 2 == 0:
                num_red = h // 2
                sum_red = num_red ** 2
                sum_blue = num_red * (num_red + 1)
            else:
                num_red = (h + 1) // 2
                sum_red = num_red ** 2
                sum_blue = ((h - 1) // 2) * ((h - 1) // 2 + 1)
        else:
            if h % 2 == 0:
                num_blue = h // 2
                sum_blue = num_blue ** 2
                sum_red = num_blue * (num_blue + 1)
            else:
                num_blue = (h + 1) // 2
                sum_blue = num_blue ** 2
                sum_red = ((h - 1) // 2) * ((h - 1) // 2 + 1)
        return sum_red, sum_blue