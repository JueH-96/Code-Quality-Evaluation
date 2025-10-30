from typing import List, Dict, Set

class Solution:
    def shortestSubstrings(self, arr: List[str]) -> List[str]:
        n = len(arr)
        
        # 1. collect the distinct substrings that every string contains
        #    and remember in which strings they appear (as a bit-mask)
        substr_to_mask: Dict[str, int] = {}
        substrs_of_string: List[Set[str]] = []
        
        for idx, s in enumerate(arr):
            seen: Set[str] = set()
            m = len(s)
            for i in range(m):
                for j in range(i + 1, m + 1):
                    sub = s[i:j]
                    seen.add(sub)
            substrs_of_string.append(seen)
            bit = 1 << idx
            for sub in seen:
                substr_to_mask[sub] = substr_to_mask.get(sub, 0) | bit
        
        # 2. for every string pick the wanted substring
        ans: List[str] = []
        
        for idx in range(n):
            own_bit = 1 << idx
            # all substrings that appear **only** in this string
            unique_subs = [sub for sub in substrs_of_string[idx]
                           if substr_to_mask[sub] == own_bit]
            if not unique_subs:
                ans.append("")
                continue
            
            # shortest length first …
            min_len = min(len(sub) for sub in unique_subs)
            candidates = [sub for sub in unique_subs if len(sub) == min_len]
            # … then lexicographically smallest
            ans.append(min(candidates))
        
        return ans