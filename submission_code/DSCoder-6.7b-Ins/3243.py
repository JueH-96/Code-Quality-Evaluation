class Solution:
    def numberOfPowerfulInt(self, start: int, finish: int, limit: int, s: str) -> int:
        powerful_ints = set()
        s_len = len(s)
        s = int(s)
        limit_pow = int(str(limit)*s_len)
        for i in range(start, min(finish+1, limit_pow+1)):
            if i % s == 0:
                if i // s <= limit:
                    powerful_ints.add(i)
        return len(powerful_ints)