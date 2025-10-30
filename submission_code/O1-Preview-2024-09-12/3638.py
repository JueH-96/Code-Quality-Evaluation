class Solution:
    def makeStringGood(self, s: str) -> int:
        counts = [0]*26
        for c in s:
            counts[ord(c)-ord('a')] +=1
        max_count = max(counts)
        min_total_cost = float('inf')
        for f in range(1, max_count+1):
            total_cost = 0
            counts_copy = counts.copy()
            # Shifting occurrences from c to c+1
            for c in range(25):  # from 'a' to 'y'
                shift = min(counts_copy[c] - f, f - counts_copy[c+1]) if counts_copy[c] > f and counts_copy[c+1] < f else 0
                counts_copy[c] -= shift
                counts_copy[c+1] += shift
                total_cost += shift  # cost per shift is 1
            # Adjust counts via deletions and insertions
            for c in range(26):
                if counts_copy[c] > f:
                    total_cost += counts_copy[c] - f  # delete extra occurrences
                elif counts_copy[c] < f:
                    total_cost += f - counts_copy[c]  # insert missing occurrences
            min_total_cost = min(min_total_cost, total_cost)
        return min_total_cost