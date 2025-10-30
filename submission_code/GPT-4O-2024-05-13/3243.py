class Solution:
    def numberOfPowerfulInt(self, start: int, finish: int, limit: int, s: str) -> int:
        count = 0
        len_s = len(s)
        
        for num in range(start, finish + 1):
            str_num = str(num)
            if str_num.endswith(s):
                if all(int(digit) <= limit for digit in str_num):
                    count += 1
        
        return count