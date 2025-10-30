class Solution:
    def minimizeStringValue(self, s: str) -> str:
        original_counts = [0] * 26
        k = 0
        # Compute the original counts and the number of '?'
        for c in s:
            if c != '?':
                original_counts[ord(c) - ord('a')] += 1
            else:
                k += 1
        
        # Step 1: Compute x[c] using the greedy approach
        x = [0] * 26
        for _ in range(k):
            min_val = float('inf')
            candidates = []
            for c in range(26):
                current = original_counts[c] + x[c]
                if current < min_val:
                    min_val = current
                    candidates = [c]
                elif current == min_val:
                    candidates.append(c)
            # Choose the lex smallest candidate
            chosen = min(candidates)
            x[chosen] += 1
        
        # Step 2: Replace '?' in the original string based on x[c]
        res = list(s)
        used = [0] * 26
        for i in range(len(res)):
            if res[i] == '?':
                min_val = float('inf')
                candidates = []
                for c in range(26):
                    if used[c] < x[c]:
                        current = original_counts[c] + used[c]
                        if current < min_val:
                            min_val = current
                            candidates = [c]
                        elif current == min_val:
                            candidates.append(c)
                # Choose the lex smallest candidate among the candidates
                chosen = min(candidates)
                res[i] = chr(ord('a') + chosen)
                used[chosen] += 1
        
        return ''.join(res)