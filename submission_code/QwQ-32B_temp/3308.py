class Solution:
    def lastNonEmptyString(self, s: str) -> str:
        from collections import defaultdict
        
        positions = defaultdict(list)
        for idx, c in enumerate(s):
            positions[c].append(idx)
        
        if not positions:
            return ""
        
        max_freq = max(len(v) for v in positions.values())
        
        target_chars = [c for c in positions if len(positions[c]) == max_freq]
        
        indices = []
        for c in target_chars:
            indices.append(positions[c][-1])
        
        indices.sort()
        
        return ''.join(s[i] for i in indices)