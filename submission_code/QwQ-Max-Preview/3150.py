class Solution:
    def shortestBeautifulSubstring(self, s: str, k: int) -> str:
        ones = [i for i, c in enumerate(s) if c == '1']
        if len(ones) < k:
            return ""
        
        min_len = float('inf')
        candidates = []
        
        for i in range(len(ones) - k + 1):
            start = ones[i]
            end = ones[i + k - 1]
            current_len = end - start + 1
            if current_len < min_len:
                min_len = current_len
                candidates = [(start, end)]
            elif current_len == min_len:
                candidates.append((start, end))
        
        if not candidates:
            return ""
        
        # Generate all possible substrings for the candidates and find the lex smallest
        substrings = [s[start:end+1] for start, end in candidates]
        return min(substrings)