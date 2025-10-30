class Solution:
    def lastNonEmptyString(self, s: str) -> str:
        # Count total occurrences of each character
        freq = [0] * 26
        for ch in s:
            freq[ord(ch) - ord('a')] += 1
        
        # The number of rounds until s becomes empty is the maximum frequency
        max_rounds = max(freq)
        
        # We want the string just before the last (max_rounds-th) round.
        # After max_rounds-1 rounds, each letter c has had its first
        # (max_rounds-1) occurrences removed.  The only letters remaining
        # are those that appear at least max_rounds times, and exactly
        # their max_rounds-th occurrence survives to that point.
        #
        # So we scan s, count occurrences, and collect each character
        # the moment it reaches its max_rounds-th occurrence.
        
        res = []
        seen = [0] * 26
        for ch in s:
            idx = ord(ch) - ord('a')
            seen[idx] += 1
            # If this is the max_rounds-th time we've seen ch,
            # and since max_rounds is the global maximum frequency,
            # this is exactly the character that survives to the last round.
            if seen[idx] == max_rounds:
                res.append(ch)
        
        return ''.join(res)