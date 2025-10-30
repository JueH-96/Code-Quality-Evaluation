from typing import List
from collections import Counter

class Solution:
    def minimumIndex(self, nums: List[int]) -> int:
        n = len(nums)
        
        # 1) Find the global dominant element and its total frequency
        freq = Counter(nums)
        dominant = max(freq, key=freq.get)     # element with highest count
        total_dom = freq[dominant]             # its total occurrences
        
        prefix_dom = 0     # occurrences of dominant element in the current prefix
        
        # 2) Scan from left to right, stopping at n-2 (suffix must be non-empty)
        for i in range(n - 1):
            if nums[i] == dominant:
                prefix_dom += 1
            
            # lengths of current prefix and suffix
            len_pref = i + 1
            len_suf  = n - len_pref
            suf_dom  = total_dom - prefix_dom  # dominant occurrences in suffix
            
            # 3) Check if dominant in both parts
            if prefix_dom * 2 > len_pref and suf_dom * 2 > len_suf:
                return i
        
        # 4) No valid split found
        return -1