class Solution:
    def beautifulNumbers(self, l: int, r: int) -> int:
        def is_beautiful(n: int) -> bool:
            digits = [int(d) for d in str(n)]
            product = 1
            summation = 0
            
            for d in digits:
                product *= d
                summation += d
            
            return product % summation == 0
        
        count = 0
        for number in range(l, r + 1):
            if is_beautiful(number):
                count += 1
        
        return count