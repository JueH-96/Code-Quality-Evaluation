class Solution:
    def hasMatch(self, s: str, p: str) -> bool:
        # Find the position of '*' in p
        star_index = p.index('*')
        
        # Split p into prefix and suffix
        prefix = p[:star_index]
        suffix = p[star_index + 1:]
        
        # Generate all possible substrings of s
        n = len(s)
        for i in range(n):
            for j in range(i + 1, n + 1):
                substring = s[i:j]
                # Check based on where '*' is in p
                if star_index == 0:
                    # '*' is at the start, check if substring ends with suffix
                    if len(suffix) <= len(substring) and substring.endswith(suffix):
                        return True
                elif star_index == len(p) - 1:
                    # '*' is at the end, check if substring starts with prefix
                    if len(prefix) <= len(substring) and substring.startswith(prefix):
                        return True
                else:
                    # '*' is in the middle, check if substring starts with prefix and ends with suffix
                    if len(prefix) + len(suffix) <= len(substring):
                        if substring.startswith(prefix) and substring.endswith(suffix):
                            return True
        return False