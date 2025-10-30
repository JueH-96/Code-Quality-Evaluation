class Solution:
    def numberOfSubstrings(self, s: str, k: int) -> int:
        def has_char_at_least_k_times(substring, k):
            from collections import Counter
            count = Counter(substring)
            for char in count:
                if count[char] >= k:
                    return True
            return False
        
        n = len(s)
        total_count = 0
        
        for i in range(n):
            for j in range(i + 1, n + 1):
                if has_char_at_least_k_times(s[i:j], k):
                    total_count += 1
        
        return total_count