class Solution:
    def maximumLength(self, s: str) -> int:
        max_ans = -1
        n = len(s)
        for c in "abcdefghijklmnopqrstuvwxyz":
            blocks = []
            i = 0
            while i < n:
                if s[i] == c:
                    j = i
                    while j < n and s[j] == c:
                        j += 1
                    blocks.append(j - i)
                    i = j
                else:
                    i += 1
            if not blocks:
                continue
            max_block = max(blocks)
            for L in range(max_block, 0, -1):
                total_occurrences = 0
                for bl in blocks:
                    if bl >= L:
                        total_occurrences += bl - L + 1
                if total_occurrences >= 3:
                    max_ans = max(max_ans, L)
                    break
        return max_ans