class Solution:
    def numberOfPowerfulInt(self, start: int, finish: int, limit: int, s: str) -> int:
        count = 0
        len_s = len(s)
        for i in range(start, finish + 1):
            num_str = str(i)
            if len(num_str) < len_s:
                continue
            if num_str[-len_s:] == s:
                is_powerful = True
                for digit in num_str:
                    if int(digit) > limit:
                        is_powerful = False
                        break
                if is_powerful:
                    count += 1
        return count