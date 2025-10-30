class Solution:
    def numberOfPowerfulInt(self, start: int, finish: int, limit: int, s: str) -> int:
        powerful_nums = set()
        s_len = len(s)
        for i in range(start, finish + 1):
            for j in range(s_len):
                if i ** (s_len - j - 1) > finish:
                    break
                powerful_nums.add(i ** (s_len - j - 1))
        return len(powerful_nums)