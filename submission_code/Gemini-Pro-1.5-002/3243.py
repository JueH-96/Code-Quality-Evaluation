class Solution:
    def numberOfPowerfulInt(self, start: int, finish: int, limit: int, s: str) -> int:
        n = len(s)
        num = int(s)
        ans = 0
        for i in range(16):
            prefix = 0
            if i > 0:
                prefix = limit * (10**i)
                if prefix + num > finish:
                    break
            
            if prefix + num >= start:
                ans += 1
            
            if i > 0:
                for j in range(1, limit + 1):
                    curr = j * (10**i) + num
                    if curr >= start and curr <= finish:
                        ans += 1

        return ans