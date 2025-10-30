class Solution:
    def makeStringGood(self, s: str) -> int:
        from collections import Counter
        counts = Counter(s)
        freqs = sorted(counts.values())
        n = len(freqs)
        ans = float('inf')
        for target_freq in set(freqs):
            curr_ops = 0
            for freq in freqs:
                curr_ops += abs(freq - target_freq)
            ans = min(ans, curr_ops)
        
        for target_freq in range(1, max(freqs) + 2):
            curr_ops = 0
            for freq in freqs:
                if freq > target_freq:
                    curr_ops += freq - target_freq
                elif freq < target_freq:
                    curr_ops += freq
            ans = min(ans, curr_ops)
        
        
        
        return ans