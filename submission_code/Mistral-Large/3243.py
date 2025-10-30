class Solution:
    def numberOfPowerfulInt(self, start: int, finish: int, limit: int, s: str) -> int:
        def is_powerful(x):
            x_str = str(x)
            if not x_str.endswith(s):
                return False
            for char in x_str:
                if int(char) > limit:
                    return False
            return True

        count = 0
        for num in range(start, finish + 1):
            if is_powerful(num):
                count += 1

        return count