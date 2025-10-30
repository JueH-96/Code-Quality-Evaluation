import itertools

class Solution:
    def minimumString(self, a: str, b: str, c: str) -> str:
        def merge_two(s: str, t: str) -> str:
            candidates = []
            # Check if one string is a substring of the other
            if t in s:
                candidates.append(s)
            if s in t:
                candidates.append(t)
            # Compute merged strings in both directions
            # Merge s followed by t
            max_overlap = 0
            for i in range(1, min(len(s), len(t)) + 1):
                if s.endswith(t[:i]):
                    max_overlap = i
            merged_st = s + t[max_overlap:]
            # Merge t followed by s
            max_overlap_rev = 0
            for i in range(1, min(len(t), len(s)) + 1):
                if t.endswith(s[:i]):
                    max_overlap_rev = i
            merged_ts = t + s[max_overlap_rev:]
            candidates.append(merged_st)
            candidates.append(merged_ts)
            # Find the minimal length candidates
            min_len = float('inf')
            best = []
            for cand in candidates:
                if len(cand) < min_len:
                    min_len = len(cand)
                    best = [cand]
                elif len(cand) == min_len:
                    best.append(cand)
            # Return the lexicographically smallest one
            return min(best)
        
        from itertools import permutations
        candidates = []
        for perm in permutations([a, b, c]):
            # Merge first two
            merged = merge_two(perm[0], perm[1])
            # Merge result with third
            merged_final = merge_two(merged, perm[2])
            candidates.append(merged_final)
        # Find the best candidate
        min_len = min(len(c) for c in candidates)
        best_candidates = [c for c in candidates if len(c) == min_len]
        return min(best_candidates)