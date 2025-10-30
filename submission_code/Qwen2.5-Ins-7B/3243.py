class Solution:
    def numberOfPowerfulInt(self, start: int, finish: int, limit: int, s: str) -> int:
        def count_numbers(n, limit, suffix):
            if n < len(suffix):
                return 0
            count = 0
            for i in range(n - len(suffix) + 1):
                prefix = str(n)[i:n - len(suffix) + 1]
                suffix_match = str(n)[n - len(suffix) + 1:]
                if suffix_match == suffix and all(int(d) <= limit for d in prefix):
                    count += 1
                elif suffix_match < suffix and all(int(d) <= limit for d in prefix):
                    count += 10 ** (len(suffix) - 1)
            return count
        
        start_count = count_numbers(start, limit, s)
        finish_count = count_numbers(finish + 1, limit, s)
        return finish_count - start_count