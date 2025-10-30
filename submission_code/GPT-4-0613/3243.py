class Solution:
    def numberOfPowerfulInt(self, start: int, finish: int, limit: int, s: str) -> int:
        count = 0
        for i in range(start, finish + 1):
            if str(i).endswith(s) and all(int(d) <= limit for d in str(i)):
                count += 1
        return count