class Solution:
    def shortestMatchingSubstring(self, s: str, p: str) -> int:
        first_star = p.find('*')
        second_star = p.find('*', first_star + 1)
        prefix = p[:first_star]
        middle = p[first_star + 1:second_star]
        suffix = p[second_star + 1:]

        min_len = float('inf')

        for i in range(len(s) - len(prefix) - len(suffix) + 1):
            if s[i:i + len(prefix)] == prefix:
                for j in range(i + len(prefix), len(s) - len(suffix) + 1):
                    if s[j:j + len(suffix)] == suffix:
                        substring = s[i:j + len(suffix)]
                        
                        middle_start = i + len(prefix)
                        middle_end = j
                        
                        if middle == "" or middle in substring[len(prefix):len(substring) - len(suffix)]:
                            min_len = min(min_len, len(substring))

        if min_len == float('inf'):
            return -1
        else:
            return min_len