class Solution:
    def minimizeStringValue(self, s: str) -> str:
        # current occurrences of every letter that is already fixed in the string
        cnt = [0] * 26
        for ch in s:
            if ch != '?':
                cnt[ord(ch) - 97] += 1

        res = list(s)

        # process the string from left to right
        for i, ch in enumerate(res):
            if ch == '?':
                # the best letter is any with the smallest current count,
                # pick the lexicographically smallest among them
                mn = min(cnt)
                for j in range(26):
                    if cnt[j] == mn:
                        res[i] = chr(j + 97)
                        cnt[j] += 1
                        break     # placed this '?', go on with the next one

        return ''.join(res)