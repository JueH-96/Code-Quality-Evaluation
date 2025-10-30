import bisect

class Solution:
    def validSequence(self, word1: str, word2: str) -> List[int]:
        n = len(word2)
        if n == 0:
            return []
        
        # Precompute for each character c, the list of indices in word1 where the character is not c
        precomputed = {}
        for c in set(word1):
            precomputed[c] = []
        for i, c in enumerate(word1):
            for key in precomputed:
                if key != c:
                    precomputed[key].append(i)
        
        # Check Case 1: zero substitutions allowed
        prev = -1
        seq = []
        case1_valid = True
        for j in range(n):
            c = word2[j]
            list_c = precomputed.get(c, [])
            idx = bisect.bisect_left(list_c, prev)
            if idx < len(list_c):
                seq.append(list_c[idx])
                prev = list_c[idx]
            else:
                case1_valid = False
                break
        if case1_valid:
            return seq
        
        # Check Case 2: one substitution allowed, used at the end
        prev = -1
        seq = []
        case2_valid = True
        for j in range(n-1):
            c = word2[j]
            list_c = precomputed.get(c, [])
            idx = bisect.bisect_left(list_c, prev)
            if idx < len(list_c):
                seq.append(list_c[idx])
                prev = list_c[idx]
            else:
                case2_valid = False
                break
        if not case2_valid:
            return []
        
        # Handle the last character for substitution
        c = word2[-1]
        list_c = precomputed.get(c, [])
        idx = bisect.bisect_left(list_c, prev)
        if idx < len(list_c):
            seq.append(list_c[idx])
            return seq
        else:
            return []
        
        return []