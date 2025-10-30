class Solution:
    def findMaximumNumber(self, k: int, x: int) -> int:
        low = 1
        high = 10**18  # A sufficiently large upper bound
        answer = 0
        
        while low <= high:
            mid = (low + high) // 2
            total = self.calculate_sum(mid, x)
            if total <= k:
                answer = mid
                low = mid + 1
            else:
                high = mid - 1
        
        return answer
    
    def calculate_sum(self, num: int, x: int) -> int:
        total = 0
        m = x  # Starting with the first position that is a multiple of x (1-based)
        while True:
            m_0 = m - 1  # Convert to 0-based index
            if (1 << m_0) > num:
                break  # This bit position is beyond the highest bit in 'num'
            
            # Calculate the number of set bits in position m_0 (0-based) from 0 to num inclusive
            higher = 1 << (m_0 + 1)
            full_cycles = (num + 1) // higher
            remainder = (num + 1) % higher
            count = full_cycles * (1 << m_0) + max(0, remainder - (1 << m_0))
            total += count
            
            m += x  # Move to the next multiple of x
        
        return total