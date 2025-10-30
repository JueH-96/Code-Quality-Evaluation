class Solution:
    def maxHeightOfTriangle(self, red: int, blue: int) -> int:
        s = red + blue
        h_candidate = 0
        while (h_candidate + 1) * (h_candidate + 2) // 2 <= s:
            h_candidate += 1
        
        for h in range(h_candidate, 0, -1):
            # Check starting with red
            m_red_start = (h + 1) // 2
            required_red = m_red_start ** 2
            m_blue_start = h // 2
            required_blue = m_blue_start * (m_blue_start + 1)
            if required_red <= red and required_blue <= blue:
                return h
            
            # Check starting with blue
            m_blue_start_b = (h + 1) // 2
            required_blue_b = m_blue_start_b ** 2
            m_red_start_b = h // 2
            required_red_b = m_red_start_b * (m_red_start_b + 1)
            if required_blue_b <= blue and required_red_b <= red:
                return h
        
        return 0  # This line is theoretically unreachable given constraints