class Solution:
    def numberOfPowerfulInt(self, start: int, finish: int, limit: int, s: str) -> int:
        def countPowerfulInt(max_val, limit, s):
            if any(int(c) > limit for c in s):
                return 0
            factor = 10**len(s)
            total_count = 0
            for length in range(1, len(s)):
                low = int('1' + '0' * (len(s) - length))
                high = int(str(limit) * length)
                total_count += min(factor - 1, high) - max(0, low - 1)
            low = int(s)
            high = int(str(min(limit, int(s[0]))) + ('9' * (len(s) - 1)))
            return total_count + (max(0, min(finish + 1, high) - max(low, start)) // factor)
        
        return countPowerfulInt(finish, limit, s) - countPowerfulInt(start - 1, limit, s)