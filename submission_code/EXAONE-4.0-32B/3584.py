import bisect
from collections import defaultdict

class Solution:
    def validSequence(self, word1: str, word2: str) -> List[int]:
        n = len(word2)
        m = len(word1)
        
        char_pos = defaultdict(list)
        for idx, char in enumerate(word1):
            char_pos[char].append(idx)
        
        def next_index(c, start):
            if c not in char_pos:
                return None
            arr = char_pos[c]
            i = bisect.bisect_left(arr, start)
            if i < len(arr):
                return arr[i]
            return None
        
        dp0 = [None] * n
        dp1 = [None] * n
        parent1 = [None] * n
        
        dp0[0] = next_index(word2[0], 0)
        all_chars = "abcdefghijklmnopqrstuvwxyz"
        candidate1_val = None
        for c in all_chars:
            pos = next_index(c, 0)
            if pos is not None:
                if candidate1_val is None or pos < candidate1_val:
                    candidate1_val = pos
        dp1[0] = candidate1_val
        
        for i in range(1, n):
            if dp0[i-1] is not None:
                dp0[i] = next_index(word2[i], dp0[i-1] + 1)
            else:
                dp0[i] = None
            
            candidates = []
            if dp0[i-1] is not None:
                cand1 = None
                for c in all_chars:
                    pos = next_index(c, dp0[i-1] + 1)
                    if pos is not None:
                        if cand1 is None or pos < cand1:
                            cand1 = pos
                if cand1 is not None:
                    candidates.append((cand1, 0))
            if dp1[i-1] is not None:
                cand2 = next_index(word2[i], dp1[i-1] + 1)
                if cand2 is not None:
                    candidates.append((cand2, 1))
            
            if candidates:
                candidates.sort(key=lambda x: x[0])
                best_val, src = candidates[0]
                dp1[i] = best_val
                parent1[i] = src
            else:
                dp1[i] = None
                parent1[i] = None
        
        candidate0_valid = (dp0[-1] is not None)
        
        candidate1_seq = None
        if dp1 and dp1[-1] is not None:
            seq = []
            state = 1
            for i in range(n-1, -1, -1):
                if state == 0:
                    seq.append(dp0[i])
                else:
                    seq.append(dp1[i])
                    if i > 0:
                        state = parent1[i]
            candidate1_seq = list(reversed(seq))
        
        if candidate0_valid and candidate1_seq is not None:
            if dp0 < candidate1_seq:
                return dp0
            else:
                return candidate1_seq
        elif candidate0_valid:
            return dp0
        elif candidate1_seq is not None:
            return candidate1_seq
        else:
            return []