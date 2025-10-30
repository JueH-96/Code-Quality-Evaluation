class Solution:
    def possibleStringCount(self, word: str) -> int:
        runs = []
        n = len(word)
        i = 0
        while i < n:
            j = i
            while j + 1 < n and word[j + 1] == word[i]:
                j += 1
            runs.append((word[i], j - i + 1))
            i = j + 1
        total_count = 1  # Start with the observed word
        for ch, length in runs:
            if length > 1:
                total_count += length - 1  # Overtyping possibilities for this run
        return total_count