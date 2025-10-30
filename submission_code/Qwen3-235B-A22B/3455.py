class Solution:
    def minimumLength(self, s: str) -> int:
        from collections import Counter
        
        cnt = Counter(s)
        n = len(s)
        
        # If all characters are the same
        if len(cnt) == 1:
            return 1 if (n % 2 == 1) else 2
        
        res = 0
        for c in cnt:
            if cnt[c] >= 2:
                res += 2
            else:
                res += 1
        return res