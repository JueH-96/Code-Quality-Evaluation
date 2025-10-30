class Solution:
    def numberOfSubstrings(self, s: str) -> int:
        count = 0
        zero_count = 0
        one_count = 0
        for ch in s:
            if ch == '0':
                zero_count += 1
            else:
                one_count += 1
                if zero_count > 0 and one_count >= zero_count**2:
                    count += zero_count + 1
                    zero_count = 0
                    one_count = 0
        if zero_count > 0 and one_count >= zero_count**2:
            count += zero_count + 1
        return count