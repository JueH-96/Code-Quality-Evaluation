class Solution:
    def maximumLength(self, s: str) -> int:
        from collections import defaultdict
        
        def is_special(sub):
            return len(set(sub)) == 1
        
        length_counts = defaultdict(list)
        i = 0
        while i < len(s):
            j = i + 1
            while j <= len(s) and is_special(s[i:j]):
                j += 1
            length_counts[s[i]].append(j - i)
            i = j
        
        max_length = -1
        for char, lengths in length_counts.items():
            lengths.sort(reverse=True)
            if len(lengths) >= 3:
                max_length = max(max_length, lengths[2])
        
        return max_length