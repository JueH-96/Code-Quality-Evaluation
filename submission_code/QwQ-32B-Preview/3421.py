from typing import List

class Solution:
    def countCompleteDayPairs(self, hours: List[int]) -> int:
        # Create frequency map for residues modulo 24
        freq = {}
        for h in hours:
            res = h % 24
            freq[res] = freq.get(res, 0) + 1
        
        count = 0
        for h in hours:
            res = h % 24
            required_res = (-res) % 24
            if required_res == res:
                # Subtract one to avoid counting the element itself
                count += (freq[required_res] - 1)
            else:
                count += freq.get(required_res, 0)
        
        # Each pair is counted twice, so divide by 2
        return count // 2