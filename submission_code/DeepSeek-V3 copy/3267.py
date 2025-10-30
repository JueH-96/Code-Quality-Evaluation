class Solution:
    def maximumLength(self, s: str) -> int:
        from collections import defaultdict

        # Dictionary to store the frequency of each special substring
        freq = defaultdict(int)

        n = len(s)
        for i in range(n):
            current_char = s[i]
            count = 1
            freq[(current_char, count)] += 1
            for j in range(i+1, n):
                if s[j] == current_char:
                    count += 1
                    freq[(current_char, count)] += 1
                else:
                    break

        max_len = -1
        for (char, length), cnt in freq.items():
            if cnt >= 3:
                if length > max_len:
                    max_len = length

        return max_len