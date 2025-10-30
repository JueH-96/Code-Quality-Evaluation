class Solution:
    def minStartingIndex(self, s: str, pattern: str) -> int:
        pattern_len = len(pattern)
        s_len = len(s)
        
        # Pattern can't fit in s
        if pattern_len > s_len:
            return -1
        
        for i in range(s_len - pattern_len + 1):
            diff_count = 0
            # Compare each character of the substring with pattern
            for j in range(pattern_len):
                if s[i + j] != pattern[j]:
                    diff_count += 1
                # Early termination if more than one difference
                if diff_count > 1:
                    break
            
            # Found a valid substring with at most one difference
            if diff_count <= 1:
                return i
        
        return -1