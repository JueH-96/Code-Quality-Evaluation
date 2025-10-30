class Solution:
    def minimizeStringValue(self, s: str) -> str:
        # Count fixed (non-'?') letters.
        fixed_count = {ch: 0 for ch in "abcdefghijklmnopqrstuvwxyz"}
        for ch in s:
            if ch != '?':
                fixed_count[ch] += 1
        
        num_q = s.count('?')
        
        # We will choose a multiset M for the '?' replacements.
        # For a letter L the cost of adding one occurrence is:
        # fixed_count[L] + (how many times we've already added L).
        added = {ch: 0 for ch in "abcdefghijklmnopqrstuvwxyz"}
        chosen = []  # multiset for replacements
        
        # For each replacement position (as an aggregate decision),
        # choose the letter with minimum incremental cost (ties -> lex smallest).
        for _ in range(num_q):
            best_letter = None
            best_cost = float('inf')
            for ch in "abcdefghijklmnopqrstuvwxyz":
                # Marginal cost if we add letter ch:
                cost = fixed_count[ch] + added[ch]
                if cost < best_cost or (cost == best_cost and (best_letter is None or ch < best_letter)):
                    best_letter = ch
                    best_cost = cost
            # Choose that letter and update.
            added[best_letter] += 1
            chosen.append(best_letter)
        
        # To obtain the lexicographically smallest final string, we distribute 
        # the replacement letters in sorted order into the '?' positions.
        chosen.sort()  # sorted list of chosen letters
        
        # Replace each '?' in s with the next letter from chosen.
        result_chars = []
        idx = 0
        for ch in s:
            if ch == '?':
                result_chars.append(chosen[idx])
                idx += 1
            else:
                result_chars.append(ch)
        
        return "".join(result_chars)