class Solution:
    def numberOfPowerfulInt(self, start: int, finish: int, limit: int, s: str) -> int:
        def is_powerful(x: int) -> bool:
            x_str = str(x)
            if not x_str.endswith(s):
                return False
            for digit in x_str:
                if int(digit) > limit:
                    return False
            return True

        count = 0
        for num in range(start, finish + 1):
            if is_powerful(num):
                count += 1

        return count