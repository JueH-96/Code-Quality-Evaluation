class Solution:
    def minimizeStringValue(self, s: str) -> str:
        counts = [0] * 26
        result = []
        for c in s:
            if c != '?':
                result.append(c)
                counts[ord(c) - ord('a')] += 1
            else:
                min_count = float('inf')
                min_char = -1
                for i in range(26):
                    if counts[i] < min_count:
                        min_count = counts[i]
                        min_char = i
                    elif counts[i] == min_count:
                        if i < min_char:
                            min_char = i
                result.append(chr(ord('a') + min_char))
                counts[min_char] += 1
        return ''.join(result)