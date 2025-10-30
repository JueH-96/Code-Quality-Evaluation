class Solution:
    def numberOfPowerfulInt(self, start: int, finish: int, limit: int, s: str) -> int:
        def count_numbers_up_to(n: int) -> int:
            if n < int(s):
                return 0
                
            n_str = str(n)
            if len(n_str) < len(s):
                return 0
                
            if not n_str.endswith(s):
                # Find largest number <= n that ends with s
                suffix_len = len(s)
                prefix = n_str[:-suffix_len]
                if int(n_str[-suffix_len:]) < int(s):
                    if not prefix:
                        return 0
                    prefix = str(int(prefix) - 1)
                n_str = prefix + s
                if not n_str or int(n_str) < start:
                    return 0
                n = int(n_str)
            
            # Now n_str ends with s
            prefix_len = len(n_str) - len(s)
            if prefix_len == 0:
                return 1 if all(int(d) <= limit for d in n_str) else 0
                
            count = 0
            prefix = n_str[:prefix_len]
            
            # Check if current number is valid
            if all(int(d) <= limit for d in n_str):
                count = 1
            
            # Count numbers with smaller prefix
            for i in range(len(prefix)):
                curr_digit = int(prefix[i])
                for d in range(curr_digit):
                    if d > limit:
                        continue
                    remaining_digits = prefix_len - i - 1
                    if remaining_digits > 0:
                        count += (min(limit + 1, 10) ** remaining_digits)
                if curr_digit > limit:
                    break
            
            return count
            
        return count_numbers_up_to(finish) - count_numbers_up_to(start - 1)