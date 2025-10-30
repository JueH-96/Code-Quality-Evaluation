from collections import Counter
from typing import List

class Solution:
    def maximumLength(self, nums: List[int]) -> int:
        count = Counter(nums)
        max_len = 0
        
        for x in count:
            current_max_m = 0
            m_candidate = 0
            
            while True:
                next_m = m_candidate + 1
                valid = True
                
                # Check all i from 0 to next_m - 1 (which is m_candidate)
                for i in range(next_m):
                    power = x ** (2 ** i)
                    if count.get(power, 0) < 2:
                        valid = False
                        break
                if not valid:
                    break
                
                # Check the next power (i = next_m)
                next_power = x ** (2 ** next_m)
                if count.get(next_power, 0) < 1:
                    break
                
                current_max_m = next_m
                m_candidate += 1
            
            current_len = 2 * current_max_m + 1
            if current_len > max_len:
                max_len = current_len
        
        return max_len