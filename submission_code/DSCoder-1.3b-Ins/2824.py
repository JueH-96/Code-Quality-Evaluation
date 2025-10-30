class Solution:
    def isFascinating(self, n: int) -> bool:
        str_n = str(n)
        str_2n = str(2 * n)
        str_3n = str(3 * n)
        
        if '0' in str_n + str_2n + str_3n:
            return False
        
        if len(set(str_n + str_2n + str_3n)) != 9:
            return False
        
        return True