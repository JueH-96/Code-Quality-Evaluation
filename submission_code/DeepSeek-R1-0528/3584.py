import bisect
from collections import defaultdict

class Solution:
    def validSequence(self, word1: str, word2: str) -> List[int]:
        n = len(word2)
        m = len(word1)
        char_pos = defaultdict(list)
        for idx, char in enumerate(word1):
            char_pos[char].append(idx)
        for key in char_pos:
            char_pos[key].sort()
        
        all_indices = list(range(m))
        
        def get_exact_match(chars, start_from):
            if not chars:
                return []
            indices = []
            current = start_from
            for c in chars:
                arr = char_pos.get(c, [])
                if not arr:
                    return None
                pos = bisect.bisect_right(arr, current)
                if pos >= len(arr):
                    return None
                indices.append(arr[pos])
                current = arr[pos]
            return indices
        
        candidates = []
        base = get_exact_match(word2, -1)
        if base is not None:
            candidates.append(base)
        
        for i in range(n + 1):
            prefix_str = word2[:i]
            prefix = get_exact_match(prefix_str, -1) if i > 0 else []
            if i > 0 and prefix is None:
                continue
            last = prefix[-1] if i > 0 else -1
            suffix_str = word2[i+1:] if i < n else ""
            if i < n:
                start_idx = bisect.bisect_right(all_indices, last)
                for j in range(start_idx, len(all_indices)):
                    p = all_indices[j]
                    suf = get_exact_match(suffix_str, p) if suffix_str else []
                    if suf is not None or not suffix_str:
                        candidate_seq = prefix + [p]
                        if suf is not None:
                            candidate_seq += suf
                        s = ''.join(word1[idx] for idx in candidate_seq)
                        mismatches = sum(1 for a, b in zip(s, word2) if a != b)
                        if mismatches <= 1:
                            candidates.append(candidate_seq)
            else:
                start_idx = bisect.bisect_right(all_indices, last)
                for j in range(start_idx, len(all_indices)):
                    p = all_indices[j]
                    candidate_seq = prefix + [p]
                    s = ''.join(word1[idx] for idx in candidate_seq)
                    mismatches = sum(1 for a, b in zip(s, word2) if a != b)
                    if mismatches <= 1:
                        candidates.append(candidate_seq)
        
        if not candidates:
            return []
        candidates.sort()
        return candidates[0]