class Solution:
    def numberOfPowerfulInt(self, start: int, finish: int, limit: int, s: str) -> int:
        def is_powerful(num_str: str) -> bool:
            return all(int(digit) <= limit for digit in num_str)
        
        def count_powerful(prefix: str, s: str, limit: int) -> int:
            if not prefix:
                return 1 if is_powerful(s) else 0
            count = 0
            for digit in range(int(prefix[0]) + 1):
                if digit == int(prefix[0]):
                    count += count_powerful(prefix[1:], s, limit)
                elif digit <= limit:
                    count += 10 ** (len(prefix) - 1)
            return count
        
        s_len = len(s)
        start_str = str(start - 1)
        finish_str = str(finish)
        
        # Count powerful numbers up to finish
        count_finish = count_powerful(finish_str[:-s_len], s, limit)
        
        # Count powerful numbers up to start - 1
        count_start = count_powerful(start_str[:-s_len], s, limit)
        
        return count_finish - count_start