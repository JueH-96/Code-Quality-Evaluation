from collections import Counter

class Solution:
    def maximumNumberOfStringPairs(self, words: List[str]) -> int:
        counts = Counter(words)
        processed = set()
        total = 0
        
        for w in list(counts.keys()):
            if w in processed:
                continue
            rev_w = w[::-1]
            if rev_w in counts and w < rev_w:
                min_pairs = min(counts[w], counts[rev_w])
                total += min_pairs
                counts[w] -= min_pairs
                counts[rev_w] -= min_pairs
                processed.add(w)
                processed.add(rev_w)
        return total