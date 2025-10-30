class Solution:
    def minimumDeletions(self, word: str, k: int) -> int:
        freq = [0] * 26
        for c in word:
            freq[ord(c) - ord('a')] += 1
        
        max_freq = max(freq)
        total_chars = sum(freq)
        min_deletions = total_chars  # Initialize with the maximum possible deletions (delete all)
        
        for m in range(1, max_freq + 1):
            current_deletions = 0
            for f in freq:
                lower_bound = max(m - k, 1)
                if f < lower_bound:
                    current_deletions += f  # Delete all occurrences of this character
                else:
                    target = min(f, m)
                    current_deletions += f - target  # Delete the difference
            if current_deletions < min_deletions:
                min_deletions = current_deletions
        
        return min_deletions