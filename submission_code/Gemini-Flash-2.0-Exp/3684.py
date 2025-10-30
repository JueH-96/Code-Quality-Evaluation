class Solution:
    def hasMatch(self, s: str, p: str) -> bool:
        star_index = p.find('*')
        prefix = p[:star_index]
        suffix = p[star_index + 1:]

        for i in range(len(s) - len(p) + 1):
            substring = s[i: i + len(prefix)]
            if substring == prefix:
                remaining_s = s[i + len(prefix):]
                if len(suffix) == 0:
                    return True
                
                for j in range(len(remaining_s) + 1):
                    potential_suffix = remaining_s[j:]
                    if len(potential_suffix) == len(suffix) and potential_suffix == suffix:
                        return True
        return False