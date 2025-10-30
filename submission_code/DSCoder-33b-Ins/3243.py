class Solution:
    def numberOfPowerfulInt(self, start: int, finish: int, limit: int, s: str) -> int:
        def is_powerful(x: int, s: str) -> bool:
            return str(x).endswith(s) and all(int(d) <= limit for d in str(x))

        return sum(is_powerful(x, s) for x in range(start, finish + 1))