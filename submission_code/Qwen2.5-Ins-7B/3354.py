from collections import Counter

class Solution:
    def minimizeStringValue(self, s: str) -> str:
        n = len(s)
        count = Counter(s.replace('?', ''))
        result = []
        for _ in range(s.count('?')):
            min_char = 'a'
            min_count = float('inf')
            for char in 'abcdefghijklmnopqrstuvwxyz':
                if count[char] < min_count:
                    min_count = count[char]
                    min_char = char
            result.append(min_char)
            count[min_char] += 1
        result.sort()
        ans = list(s)
        for i, char in enumerate(result):
            ans[s.index('?', start=i) if '?' in s[i:] else len(ans)] = char
        return ''.join(ans)