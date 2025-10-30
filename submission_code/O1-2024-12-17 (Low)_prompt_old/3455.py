class Solution:
    def minimumLength(self, s: str) -> int:
        from collections import Counter
        
        freq = Counter(s)
        ans = 0
        
        for char, count in freq.items():
            if count < 3:
                # If there are fewer than 3 occurrences, none can be removed.
                ans += count
            else:
                # If we have at least 3 occurrences:
                #  - leftover is 1 if count is odd
                #  - leftover is 2 if count is even
                ans += 1 if count % 2 == 1 else 2
        
        return ans