class Solution:
    def numberOfPowerfulInt(self, start: int, finish: int, limit: int, s: str) -> int:
        def atMost(n):
            res = 0
            s = str(n)
            m = len(s)
            k = len(s) - len(s) % len(s)
            for i in range(k, m + 1, len(s)):
                prefix = s[:i]
                if prefix[0] > str(limit):
                    break
                if prefix[0] == str(limit):
                    prefix = prefix[:-1] + str(int(prefix[-1]) + 1)
                res += int(prefix) // int(s)
            return res
        
        return atMost(finish) - atMost(start - 1)