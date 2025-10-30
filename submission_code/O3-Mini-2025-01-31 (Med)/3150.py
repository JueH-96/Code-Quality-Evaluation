class Solution:
    def shortestBeautifulSubstring(self, s: str, k: int) -> str:
        # First, collect indices where s has '1'
        one_indices = [i for i, ch in enumerate(s) if ch == '1']
        if len(one_indices) < k:
            return ""
        
        min_len = float('inf')
        candidates = []
        
        # For each contiguous block of k ones, the minimal substring that
        # contains exactly k ones is from the first to the kth one.
        for i in range(len(one_indices) - k + 1):
            start = one_indices[i]
            end = one_indices[i + k - 1]
            current_len = end - start + 1
            
            # Only consider substrings of minimal length (based on k ones group)
            if current_len < min_len:
                min_len = current_len
                candidates = [s[start:end + 1]]
            elif current_len == min_len:
                candidates.append(s[start:end + 1])
        
        # Return the lexicographically smallest candidate among those with minimal length.
        return min(candidates)