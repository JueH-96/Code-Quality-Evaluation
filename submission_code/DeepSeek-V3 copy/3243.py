class Solution:
    def numberOfPowerfulInt(self, start: int, finish: int, limit: int, s: str) -> int:
        s_len = len(s)
        s_num = int(s)
        count = 0
        
        # Generate all possible prefixes
        # The prefix can be any number with digits <= limit, and when concatenated with s, it should be within [start, finish]
        # The length of the prefix can be from 0 to the maximum possible
        
        # Determine the maximum length of the prefix
        # The total number of digits in the powerful integer is prefix_len + s_len
        # The powerful integer must be >= start and <= finish
        
        # Find the minimum and maximum possible powerful integers
        min_powerful = s_num
        max_powerful = int('9' * (s_len)) if s_len > 0 else 0
        
        # Adjust min_powerful and max_powerful based on the limit
        # The digits in the prefix must be <= limit
        # So the maximum prefix is a number with all digits as limit
        
        # The maximum prefix is a number with all digits as limit, and the length is such that the total number is <= finish
        # The total number is prefix * 10^s_len + s_num
        
        # To find the maximum prefix length, we can iterate from 0 to the maximum possible length
        # For each length, we calculate the maximum prefix and check if the resulting number is <= finish
        
        # First, find the maximum possible length of the prefix
        max_prefix_len = 0
        while True:
            # The total number of digits is max_prefix_len + s_len
            # The maximum number is (limit * 10^{max_prefix_len-1} + ... + limit) * 10^s_len + s_num
            # Which is limit * (10^{max_prefix_len} - 1) / 9 * 10^s_len + s_num
            # We need to find the maximum max_prefix_len such that this number <= finish
            # We can compute this iteratively
            if max_prefix_len == 0:
                total = s_num
            else:
                prefix_max = int(str(limit) * max_prefix_len)
                total = prefix_max * (10 ** s_len) + s_num
            if total > finish:
                break
            max_prefix_len += 1
        max_prefix_len -= 1
        
        # Now, for each prefix length from 0 to max_prefix_len, count the valid prefixes
        for prefix_len in range(0, max_prefix_len + 1):
            if prefix_len == 0:
                # Only the suffix itself
                if s_num >= start and s_num <= finish:
                    count += 1
                continue
            # The prefix can be any number with prefix_len digits, all digits <= limit
            # The total number is prefix * 10^s_len + s_num
            # We need to find the range of prefix such that the total number is >= start and <= finish
            # So, prefix * 10^s_len + s_num >= start => prefix >= (start - s_num) / 10^s_len
            # prefix * 10^s_len + s_num <= finish => prefix <= (finish - s_num) / 10^s_len
            lower_bound = max(0, (start - s_num + (10 ** s_len - 1)) // (10 ** s_len))
            upper_bound = (finish - s_num) // (10 ** s_len)
            # The prefix must be a number with prefix_len digits, all digits <= limit
            # So, the minimum prefix is 10^{prefix_len-1} if prefix_len > 0, else 0
            # The maximum prefix is limit * (10^{prefix_len} - 1) / 9
            min_prefix = 10 ** (prefix_len - 1) if prefix_len > 0 else 0
            max_prefix = int(str(limit) * prefix_len)
            # The valid prefix range is [max(lower_bound, min_prefix), min(upper_bound, max_prefix)]
            valid_lower = max(lower_bound, min_prefix)
            valid_upper = min(upper_bound, max_prefix)
            if valid_lower > valid_upper:
                continue
            # The count is valid_upper - valid_lower + 1
            count += valid_upper - valid_lower + 1
        return count