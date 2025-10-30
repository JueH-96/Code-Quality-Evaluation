class Solution:
    def minimumDeletions(self, word: str, k: int) -> int:
        from collections import Counter
        freq = Counter(word)
        freq_list = list(freq.values())
        freq_list.sort(reverse=True)
        
        min_deletions = float('inf')
        n = len(freq_list)
        
        for i in range(n):
            f_max = freq_list[i]
            f_min = f_max - k
            deletions = 0
            for f in freq_list:
                if f > f_max:
                    deletions += f - f_max
                elif f < f_min:
                    deletions += f
            min_deletions = min(min_deletions, deletions)
        
        return min_deletions