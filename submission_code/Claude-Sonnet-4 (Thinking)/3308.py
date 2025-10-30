class Solution:
    def lastNonEmptyString(self, s: str) -> str:
        from collections import Counter
        
        count = Counter(s)
        max_freq = max(count.values())
        
        # Find characters with maximum frequency
        max_chars = {c for c, freq in count.items() if freq == max_freq}
        
        # Find the last occurrence of each max frequency character
        result = []
        
        for i in range(len(s) - 1, -1, -1):
            if s[i] in max_chars:
                result.append(s[i])
                max_chars.remove(s[i])
                if not max_chars:
                    break
        
        return ''.join(reversed(result))