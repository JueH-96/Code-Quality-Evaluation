class Solution:
    def minimizeStringValue(self, s: str) -> str:
        from collections import defaultdict

        initial_counts = defaultdict(int)
        for c in s:
            initial_counts[c] += 1

        q = s.count('?')

        sorted_chars = sorted(initial_counts.keys(), key=lambda x: initial_counts[x])

        r = q % 26
        base = q // 26

        added = defaultdict(int)
        for i, c in enumerate(sorted_chars):
            added[c] = base + (1 if i < r else 0)

        res = list(s)
        for i in range(len(res)):
            if res[i] == '?':
                for c in 'abcdefghijklmnopqrstuvwxyz':
                    if added[c] > 0:
                        res[i] = c
                        added[c] -= 1
                        break

        return ''.join(res)