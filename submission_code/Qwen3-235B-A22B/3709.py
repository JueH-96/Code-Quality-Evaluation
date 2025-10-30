class Solution:
    def hasSpecialSubstring(self, s: str, k: int) -> bool:
        if not s:
            return False
        
        runs = []
        prev_char = s[0]
        start = 0
        
        for i in range(1, len(s)):
            if s[i] != prev_char:
                end = i - 1
                runs.append((prev_char, start, end))
                prev_char = s[i]
                start = i
        
        # Add the last run
        runs.append((prev_char, start, len(s) - 1))
        
        for run in runs:
            char, start_idx, end_idx = run
            run_length = end_idx - start_idx + 1
            if run_length != k:
                continue
            
            before_ok = True
            after_ok = True
            
            if start_idx > 0:
                if s[start_idx - 1] == char:
                    before_ok = False
            
            if end_idx < len(s) - 1:
                if s[end_idx + 1] == char:
                    after_ok = False
            
            if before_ok and after_ok:
                return True
        
        return False