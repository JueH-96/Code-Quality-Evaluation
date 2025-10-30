class Solution:
    def lastNonEmptyString(self, s: str) -> str:
        from collections import Counter
        
        # Count frequencies of each character
        freq = Counter(s)
        
        # The number of passes needed is the maximum frequency
        max_freq = max(freq.values())
        
        # If the maximum frequency is 1, then the string is removed in one pass
        # so the last non-empty string is the original string
        if max_freq == 1:
            return s
        
        # Otherwise, we want the string after (max_freq - 1) passes.
        # That means we remove the first (max_freq - 1) occurrences of each character.
        remove_limit = max_freq - 1
        
        seen = Counter()
        result = []
        
        for ch in s:
            seen[ch] += 1
            # If we have seen this character <= remove_limit times so far,
            # it belongs to the ones we remove first. Otherwise, keep it.
            if seen[ch] > remove_limit:
                result.append(ch)
        
        return "".join(result)