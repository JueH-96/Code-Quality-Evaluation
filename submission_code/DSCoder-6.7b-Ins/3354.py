class Solution:
    def minimizeStringValue(self, s: str) -> str:
        s = list(s)
        n = len(s)
        prefix = [0]*n
        suffix = [0]*n
        for i in range(1, n):
            if s[i-1] == s[i]:
                prefix[i] = prefix[i-1] + 1
            else:
                prefix[i] = prefix[i-1]
        for i in range(n-2, -1, -1):
            if s[i] == s[i+1]:
                suffix[i] = suffix[i+1] + 1
            else:
                suffix[i] = suffix[i+1]
        min_val = float('inf')
        min_str = ''
        for c in range(26):
            ch = chr(ord('a')+c)
            total = 0
            for i in range(n):
                if s[i] == '?':
                    left = prefix[i] if i > 0 else 0
                    right = suffix[i] if i < n-1 else 0
                    total = total + min(left, right)
            if total < min_val:
                min_val = total
                min_str = ''.join(s)
            elif total == min_val:
                min_str = min(min_str, ''.join(s))
            for i in range(n):
                if s[i] == '?':
                    s[i] = ch
        return min_str