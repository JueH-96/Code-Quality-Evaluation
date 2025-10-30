class Solution:
    def countSymmetricIntegers(self, low: int, high: int) -> int:
        count = 0
        
        def is_symmetric(num):
            s = str(num)
            length = len(s)
            if length % 2 != 0:
                return False
            
            half_len = length // 2
            sum1 = sum(int(s[i]) for i in range(half_len))
            sum2 = sum(int(s[i]) for i in range(half_len, length))

            return sum1 == sum2
        
        for i in range(low, high + 1):
            if is_symmetric(i):
                count += 1
        
        return count