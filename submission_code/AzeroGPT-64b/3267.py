from collections import Counter
class Solution:
    def maximumLength(self, s: str) -> int:
        def get_valid_substrings(sub):
            if len(set(sub)) > 1:
                return set()
            return {sub[i:j] for i in range(len(sub)) for j in range(i+1, len(sub)+1)}
        
        substring_count = Counter()
        
        for i in range(len(s)):
            for j in range(i+1, len(s)+1):
                substring = s[i:j]
                substring_count.update(get_valid_substrings(substring))

        thrice_substrings = {k:v for k, v in substring_count.items() if v >= 3 and len(set(k)) == 1}
        if not thrice_substrings:
            return -1

        return max([len(k) for k, v in thrice_substrings.items()])