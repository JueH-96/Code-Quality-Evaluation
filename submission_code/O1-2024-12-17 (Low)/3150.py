class Solution:
    def shortestBeautifulSubstring(self, s: str, k: int) -> str:
        n = len(s)
        min_len = float('inf')
        res = ""
        
        # We'll try all substrings and keep track of the ones that have exactly k '1's.
        for start in range(n):
            count_ones = 0
            for end in range(start, n):
                if s[end] == '1':
                    count_ones += 1
                # If we've found a substring with exactly k ones, check if it's the shortest
                if count_ones == k:
                    curr_len = end - start + 1
                    curr_sub = s[start:end+1]
                    if curr_len < min_len:
                        min_len = curr_len
                        res = curr_sub
                    elif curr_len == min_len and curr_sub < res:
                        res = curr_sub
        
        return res if min_len != float('inf') else ""