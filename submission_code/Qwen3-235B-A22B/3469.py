class Solution:
    def maxHeightOfTriangle(self, red: int, blue: int) -> int:
        total = red + blue
        h = 0
        while (h + 1) * (h + 2) // 2 <= total:
            h += 1
        
        for h_candidate in range(h, 0, -1):
            k_odd = (h_candidate + 1) // 2
            sum_odd = k_odd * k_odd
            m_even = h_candidate // 2
            sum_even = m_even * (m_even + 1)
            
            if (sum_odd <= red and sum_even <= blue) or (sum_even <= red and sum_odd <= blue):
                return h_candidate
        return 0