class Solution:
    def numberOfPowerfulInt(self, start: int, finish: int, limit: int, s: str) -> int:
        def dfs(index, num):
            if num > finish:
                return 0
            if str(num).endswith(s):
                return 1
            count = 0
            for i in range(limit + 1):
                if index == 0 and i == 0:
                    continue
                count += dfs(index + 1, num * 10 + i)
            return count

        count = 0
        for i in range(start, finish + 1):
            if str(i).endswith(s) and all(int(digit) <= limit for digit in str(i)):
                count += 1
        return count