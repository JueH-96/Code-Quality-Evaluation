from itertools import permutations

class Solution:
    def minimumString(self, a: str, b: str, c: str) -> str:
        def merge(s, t):
            # Compute merged1: s followed by t with maximum possible overlap
            merged1 = s + t
            for k in range(min(len(s), len(t)), 0, -1):
                if s.endswith(t[:k]):
                    merged1 = s + t[k:]
                    break
            
            # Compute merged2: t followed by s with maximum possible overlap
            merged2 = t + s
            for k in range(min(len(t), len(s)), 0, -1):
                if t.endswith(s[:k]):
                    merged2 = t + s[k:]
                    break
            
            # Choose the best between merged1 and merged2
            candidates = [merged1, merged2]
            min_len = min(len(c) for c in candidates)
            possible = [c for c in candidates if len(c) == min_len]
            return min(possible, key=lambda x: x)
        
        candidates = set()
        
        # Generate all permutations and merge
        for perm in permutations([a, b, c]):
            merged = merge(merge(perm[0], perm[1]), perm[2])
            candidates.add(merged)
        
        # Check if any of the original strings is a valid candidate
        for s in (a, b, c):
            if (a in s) and (b in s) and (c in s):
                candidates.add(s)
        
        # Find the minimal candidate
        min_length = min(len(c) for c in candidates)
        possible_candidates = [c for c in candidates if len(c) == min_length]
        return min(possible_candidates, key=lambda x: x)