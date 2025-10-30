class Solution:
    def minimumLength(self, s: str) -> int:
        # Count frequency of each character
        freq = [0] * 26
        for ch in s:
            freq[ord(ch) - ord('a')] += 1

        res = 0
        for cnt in freq:
            if cnt <= 2:
                # If there are 0, 1, or 2 occurrences, we cannot reduce below cnt
                res += cnt
            else:
                # If more than 2 occurrences:
                #   each operation removes 2 occurrences
                #   we can repeat until cnt < 3
                #   so the final remainder is:
                #     1 if cnt is odd, 2 if cnt is even
                if cnt % 2 == 1:
                    res += 1
                else:
                    res += 2
        return res