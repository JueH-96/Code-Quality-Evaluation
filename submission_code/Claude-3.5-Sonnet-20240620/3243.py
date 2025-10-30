class Solution:
    def numberOfPowerfulInt(self, start: int, finish: int, limit: int, s: str) -> int:
        def count(n):
            if n < int(s):
                return 0
            
            digits = str(n)
            if int(digits[-len(s):]) < int(s):
                return 0
            
            if len(digits) == len(s):
                return 1
            
            prefix_len = len(digits) - len(s)
            prefix = digits[:prefix_len]
            
            if all(int(d) <= limit for d in prefix):
                return int(prefix) + 1
            
            result = 0
            for i in range(prefix_len):
                if int(digits[i]) < limit:
                    result += (limit + 1) ** (prefix_len - i - 1)
                elif int(digits[i]) == limit:
                    continue
                else:
                    break
            
            return result
        
        return count(finish) - count(start - 1)