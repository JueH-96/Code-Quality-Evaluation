class Solution:
    def maxSubstringLength(self, s: str, k: int) -> bool:
        if k == 0:
            return True
        
        n = len(s)
        first_occ = {}
        last_occ = {}
        for i, char in enumerate(s):
            if char not in first_occ:
                first_occ[char] = i
            last_occ[char] = i
        
        intervals = set()
        for c in first_occ:
            l = first_occ[c]
            r = last_occ[c]
            changed = True
            while changed:
                changed = False
                new_l = l
                new_r = r
                if l == 0 and r == n - 1:
                    break
                for idx in range(l, r + 1):
                    char_in = s[idx]
                    if first_occ[char_in] < new_l:
                        new_l = first_occ[char_in]
                        changed = True
                    if last_occ[char_in] > new_r:
                        new_r = last_occ[char_in]
                        changed = True
                l = new_l
                r = new_r
            
            if not (l == 0 and r == n - 1):
                intervals.add((l, r))
        
        intervals_sorted = sorted(intervals, key=lambda x: x[1])
        count = 0
        current_right = -1
        for l, r in intervals_sorted:
            if l > current_right:
                count += 1
                current_right = r
        
        return count >= k