class Solution:
    def numberOfPowerfulInt(self, start: int, finish: int, limit: int, s: str) -> int:
        def countPowerfulIntegers(x):
            str_x = str(x)
            len_s = len(s)
            if len(str_x) < len_s:
                return 0
            count = 0
            for i in range(len(str_x) - len_s + 1):
                if str_x[i:i+len_s] == s:
                    if all(int(digit) <= limit for digit in str_x[:i]):
                        count += 1
            return count

        return countPowerfulIntegers(finish) - countPowerfulIntegers(start - 1)