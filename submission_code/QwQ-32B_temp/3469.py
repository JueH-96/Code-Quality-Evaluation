class Solution:
    def maxHeightOfTriangle(self, red: int, blue: int) -> int:
        total = red + blue
        h_max = 0
        current_sum = 0
        h = 0  # Initialize h to 0 so that the first increment makes it 1
        
        while True:
            h += 1
            current_sum += h
            if current_sum > total:
                break
            h_max = h
        
        # Now check from h_max down to 1
        for h_candidate in range(h_max, 0, -1):
            m_red = (h_candidate + 1) // 2
            k_blue = h_candidate // 2
            
            # Scenario starting with red
            sum_red_r = m_red * m_red
            sum_blue_r = k_blue * (k_blue + 1)
            if sum_red_r <= red and sum_blue_r <= blue:
                return h_candidate
            
            # Scenario starting with blue
            sum_blue_b = m_red * m_red
            sum_red_b = k_blue * (k_blue + 1)
            if sum_blue_b <= blue and sum_red_b <= red:
                return h_candidate
        
        return 0  # This line is theoretically unreachable due to problem constraints