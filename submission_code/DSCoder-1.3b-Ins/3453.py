class Solution:
    def validStrings(self, n: int) -> List[str]:
        if n == 1:
            return ["0", "1"]
        
        result = []
        for i in range(2**(n-1)):
            binary = bin(i)[2:].zfill(n-1)
            if '11' not in binary:
                result.append('0' + binary + '1')
        return result