class Solution:
    def maxSubstringLength(self, s: str, k: int) -> bool:
        if k == 0:
            return True
        
        n = len(s)
        first_occ = {}
        last_occ = {}
        for idx, char in enumerate(s):
            if char not in first_occ:
                first_occ[char] = idx
            last_occ[char] = idx
        
        count = 0
        i = 0
        while i < n and count < k:
            R = -1
            found_invalid = False
            found_segment = False
            j = i
            while j < n:
                if first_occ[s[j]] < i:
                    found_invalid = True
                    break
                if last_occ[s[j]] > R:
                    R = last_occ[s[j]]
                if j >= R:
                    if i == 0 and j == n - 1:
                        pass
                    else:
                        found_segment = True
                    break
                j += 1
            if found_invalid or j == n:
                i += 1
            elif found_segment:
                count += 1
                i = j + 1
            else:
                i += 1
        
        return count >= k