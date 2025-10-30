class Solution:
    def minimizeStringValue(self, s: str) -> str:
        count = [0] * 26
        res = []
        for c in s:
            if c != '?':
                res.append(c)
                count[ord(c) - ord('a')] += 1
            else:
                min_count = min(count)
                for i in range(26):
                    if count[i] == min_count:
                        res.append(chr(ord('a') + i))
                        count[i] += 1
                        break
        return ''.join(res)