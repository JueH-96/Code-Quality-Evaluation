from itertools import permutations

class Solution:
    def minimumString(self, a: str, b: str, c: str) -> str:
        def max_overlap(s, t):
            max_possible = min(len(s), len(t))
            for k in range(max_possible, 0, -1):
                if s.endswith(t[:k]):
                    return k
            return 0

        def merge(s, t):
            overlap_st = max_overlap(s, t)
            option1 = s + t[overlap_st:]
            
            overlap_ts = max_overlap(t, s)
            option2 = t + s[overlap_ts:]
            
            if len(option1) < len(option2):
                return option1
            elif len(option2) < len(option1):
                return option2
            else:
                return min(option1, option2)

        perms = permutations([a, b, c])
        candidates = []

        for perm in perms:
            x, y, z = perm
            s = merge(x, y)
            result = merge(s, z)
            candidates.append(result)
        
        if b in a and c in a:
            candidates.append(a)
        if a in b and c in b:
            candidates.append(b)
        if a in c and b in c:
            candidates.append(c)
        
        if not candidates:
            return ''
        
        candidates.sort(key=lambda x: (len(x), x))
        return candidates[0]