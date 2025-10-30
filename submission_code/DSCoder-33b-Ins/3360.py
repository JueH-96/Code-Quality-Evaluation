class Solution:
    def minimumDeletions(self, word: str, k: int) -> int:
        from collections import Counter
        freq = Counter(word)
        freq_values = list(freq.values())
        freq_values.sort(reverse=True)
        i = 0
        while i < len(freq_values) - 1:
            if freq_values[i] <= freq_values[i+1]:
                i += 1
            elif freq_values[i] > freq_values[i+1] + k:
                freq_values[i] = freq_values[i+1] + k
                i += 1
            else:
                freq_values[i+1] = freq_values[i] - 1
                i += 1
        return len(word) - sum(freq_values)