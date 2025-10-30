class Solution:
    def numberOfPowerfulInt(self, start: int, finish: int, limit: int, s: str) -> int:
        def count_up_to(n: int) -> int:
            s_len = len(s)
            if n < int(s):
                return 0
            
            n_str = str(n)
            prefix_len = len(n_str) - s_len
            
            if prefix_len == 0:
                return 1
            
            count = 0
            for i in range(prefix_len):
                if i == 0:
                    max_digit = min(int(n_str[i]), limit)
                else:
                    max_digit = limit
                
                if i == prefix_len - 1:
                    count += max_digit + 1
                else:
                    count += max_digit * (limit + 1)
            
            prefix = int(n_str[:prefix_len])
            if prefix <= limit:
                if n_str[prefix_len:] >= s:
                    count += 1
            
            return count
        
        return count_up_to(finish) - count_up_to(start - 1)