from itertools import permutations

class Solution:
    def minimumString(self, a: str, b: str, c: str) -> str:
        # helper that returns the shortest (and then lexicographically smallest)
        # string that contains s and t as substrings
        def merge(s: str, t: str) -> str:
            # if one already contains the other we are done
            if t in s:
                return s
            if s in t:
                return t
            
            best = s + t      # worst-case, no overlap
            
            # overlap: suffix of s with prefix of t
            for l in range(min(len(s), len(t)), 0, -1):
                if s[-l:] == t[:l]:
                    cand = s + t[l:]
                    if len(cand) < len(best) or (len(cand) == len(best) and cand < best):
                        best = cand
                    break      # longer l → shorter string, no need to keep checking
            
            # overlap: suffix of t with prefix of s
            for l in range(min(len(s), len(t)), 0, -1):
                if t[-l:] == s[:l]:
                    cand = t + s[l:]
                    if len(cand) < len(best) or (len(cand) == len(best) and cand < best):
                        best = cand
                    break
            
            return best
        
        best_ans = None
        for p in permutations([a, b, c]):
            cur = merge(merge(p[0], p[1]), p[2])
            if best_ans is None or len(cur) < len(best_ans) or (len(cur) == len(best_ans) and cur < best_ans):
                best_ans = cur
        return best_ans