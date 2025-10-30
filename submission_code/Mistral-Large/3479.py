class Solution:
    def numberOfSubstrings(self, s: str) -> int:
        n = len(s)
        dominant_count = 0

        # Iterate over all possible substrings
        for i in range(n):
            zero_count = 0
            one_count = 0
            for j in range(i, n):
                if s[j] == '0':
                    zero_count += 1
                else:
                    one_count += 1

                if one_count >= zero_count * zero_count:
                    dominant_count += 1

        return dominant_count