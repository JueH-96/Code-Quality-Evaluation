class Solution:
    def numberOfPowerfulInt(self, start: int, finish: int, limit: int, s: str) -> int:
        def count_powerful(n):
            if n < int(s):
                return 0
            
            n_str = str(n)
            s_len = len(s)
            n_len = len(n_str)
            
            # If n is shorter than s, no valid numbers
            if n_len < s_len:
                return 0
            
            # Number of prefix digits
            prefix_len = n_len - s_len
            
            # If prefix_len is 0, we just need to check if n >= int(s)
            if prefix_len == 0:
                return 1 if n >= int(s) else 0
            
            count = 0
            
            # Count numbers with fewer total digits
            for total_digits in range(s_len, n_len):
                prefix_digits = total_digits - s_len
                if prefix_digits == 0:
                    count += 1
                else:
                    # First digit can be 1 to limit, rest can be 0 to limit
                    count += (limit) * (limit + 1) ** (prefix_digits - 1)
            
            # Count numbers with same total digits as n
            n_prefix = n_str[:prefix_len]
            
            # Count prefixes that are lexicographically smaller
            for i in range(prefix_len):
                digit = int(n_prefix[i])
                
                # Count valid digits smaller than current digit
                if i == 0:
                    # First digit: can be 1 to min(digit-1, limit)
                    upper = min(digit - 1, limit)
                    if upper >= 1:
                        count += upper * (limit + 1) ** (prefix_len - 1 - i)
                else:
                    # Other digits: can be 0 to min(digit-1, limit)
                    upper = min(digit - 1, limit)
                    if upper >= 0:
                        count += (upper + 1) * (limit + 1) ** (prefix_len - 1 - i)
                
                # If current digit exceeds limit, we can't continue
                if digit > limit:
                    return count
            
            # Check if the exact number n is valid
            # All prefix digits must be <= limit, and n must end with s
            valid_prefix = all(int(d) <= limit for d in n_prefix)
            ends_with_s = n_str.endswith(s)
            
            if valid_prefix and ends_with_s:
                count += 1
            
            return count
        
        return count_powerful(finish) - count_powerful(start - 1)