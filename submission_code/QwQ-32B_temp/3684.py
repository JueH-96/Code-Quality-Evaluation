class Solution:
    def hasMatch(self, s: str, p: str) -> bool:
        star_pos = p.find('*')
        prefix = p[:star_pos]
        suffix = p[star_pos+1:]
        len_prefix = len(prefix)
        len_suffix = len(suffix)
        total_min = len_prefix + len_suffix
        
        for i in range(0, len(s) - len_prefix + 1):
            if s[i:i+len_prefix] == prefix:
                remaining = s[i + len_prefix:]
                if suffix in remaining:
                    if total_min == 0:
                        # The replaced * must be non-empty (remaining must be non-empty)
                        if len(remaining) >= 1:
                            return True
                    else:
                        return True
        return False