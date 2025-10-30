class Solution:
    def hasSpecialSubstring(self, s: str, k: int) -> bool:
        n = len(s)
        i = 0
        while i < n:
            # Find the end of the run starting at i
            j = i
            while j < n and s[j] == s[i]:
                j += 1
            run_length = j - i
            if run_length == k:
                return True
            # Move to the start of the next potential run
            i = j
        return False