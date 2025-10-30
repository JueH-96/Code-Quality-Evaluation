class Solution:
    def hasMatch(self, s: str, p: str) -> bool:
        # Find the position of '*' in p
        star_index = p.index('*')
        before = p[:star_index]
        after = p[star_index + 1:]
        
        # Handle different cases based on before and after
        if before == "" and after == "":
            return False  # Shouldn't happen per constraints
        elif before == "":
            # Check if after is in s
            return after in s
        elif after == "":
            # Check if before is in s
            return before in s
        else:
            # Both before and after are non-empty
            len_before = len(before)
            len_after = len(after)
            for i in range(len(s) - len_before + 1):
                if s[i:i + len_before] == before:
                    # Look for after starting from i + len_before to end of s
                    for j in range(i + len_before, len(s) - len_after + 1):
                        if s[j:j + len_after] == after:
                            return True
            return False