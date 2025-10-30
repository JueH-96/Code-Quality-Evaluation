class Solution:
    def minimumDeletions(self, word: str, k: int) -> int:
        # Count frequencies of each character (only 26 lowercase letters)
        freq = [0]*26
        for ch in word:
            freq[ord(ch) - ord('a')] += 1
        
        max_freq = max(freq)
        best_cost = float('inf')
        
        # Try all possible "minimum frequencies" x from 0 up to max_freq
        # Kept frequencies must then lie in [x, x+k].
        for x in range(max_freq + 1):
            cost = 0
            upper_bound = x + k
            for f in freq:
                if f < x:
                    # We can't raise f to x (can't add letters), so remove them all
                    cost += f
                elif f > upper_bound:
                    # We can only keep up to upper_bound
                    cost += (f - upper_bound)
            best_cost = min(best_cost, cost)
        
        return best_cost