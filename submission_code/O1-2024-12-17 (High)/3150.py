class Solution:
    def shortestBeautifulSubstring(self, s: str, k: int) -> str:
        n = len(s)
        # Prefix array to count number of '1's up to index i
        prefix_ones = [0] * (n + 1)
        for i in range(n):
            prefix_ones[i + 1] = prefix_ones[i] + (s[i] == '1')
        
        min_len = float('inf')
        answer = ""
        
        # Enumerate all substrings
        for start in range(n):
            for end in range(start + 1, n + 1):
                count_ones = prefix_ones[end] - prefix_ones[start]
                if count_ones == k:
                    length = end - start
                    if length < min_len:
                        min_len = length
                        answer = s[start:end]
                    elif length == min_len:
                        candidate = s[start:end]
                        if candidate < answer:
                            answer = candidate
        
        return answer if answer else ""