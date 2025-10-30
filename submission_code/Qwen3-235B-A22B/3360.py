class Solution:
    def minimumDeletions(self, word: str, k: int) -> int:
        from collections import Counter
        
        cnt = Counter(word)
        freqs = list(cnt.values())
        if not freqs:
            return 0  # As per problem constraints, word is non-empty
        
        max_freq = max(freqs)
        original_total = sum(freqs)
        max_sum = 0
        
        for M in range(0, max_freq + 1):
            current_sum = 0
            if M == 0:
                current_sum = 0  # This is deleting all characters
            else:
                L = max(1, M - k)
                for f in freqs:
                    if f >= L:
                        current_sum += min(f, M)
            # Update max_sum if current_sum is larger
            if current_sum > max_sum:
                max_sum = current_sum
        
        return original_total - max_sum