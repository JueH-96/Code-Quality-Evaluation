class Solution:
    def minimumLength(self, s: str) -> int:
        # We count how many times each character appears.
        # Then, for each character c with frequency freq[c]:
        #  - If freq[c] = 0, it contributes 0 to the final length.
        #  - If freq[c] is odd (≥ 1), it contributes 1.
        #  - If freq[c] is even (≥ 2), it contributes 2.
        #
        # This works because each operation removes pairs of the same character
        # around a "middle" occurrence that remains in place. As long as there
        # are at least three occurrences of a character, we can remove two of
        # them. Eventually, for each character:
        #  - An odd count can be reduced to 1,
        #  - An even count can be reduced to 2 (if it's at least 2).

        freq = [0]*26  # since the string consists of lowercase English letters
        for ch in s:
            freq[ord(ch) - ord('a')] += 1

        ans = 0
        for c in freq:
            if c == 0:
                continue
            if c % 2 == 1:  # odd frequency
                ans += 1
            else:           # even frequency
                ans += 2
        return ans