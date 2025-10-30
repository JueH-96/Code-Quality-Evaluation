class Solution:
    def maximumLength(self, s: str) -> int:
        def count_occurrences(s, sub):
            count = 0
            for i in range(len(s) - len(sub) + 1):
                if s[i:i+len(sub)] == sub:
                    count += 1
            return count

        unique_chars = set(s)
        max_length = -1

        for k in range(1, len(s) + 1):
            for c in unique_chars:
                p = c * k
                cnt = count_occurrences(s, p)
                if cnt >= 3:
                    max_length = max(max_length, k)

        return max_length if max_length != -1 else -1