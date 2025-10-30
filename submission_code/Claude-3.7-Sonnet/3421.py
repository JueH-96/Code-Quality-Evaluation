class Solution:
    def countCompleteDayPairs(self, hours: List[int]) -> int:
        remainders = {}
        count = 0
        
        for hour in hours:
            r = hour % 24
            complement = (24 - r) % 24
            
            if complement in remainders:
                count += remainders[complement]
                
            remainders[r] = remainders.get(r, 0) + 1
        
        return count