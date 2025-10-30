class Solution:
    def minimumDeletions(self, word: str, k: int) -> int:
        char_counts = {}
        for char in word:
            char_counts[char] = char_counts.get(char, 0) + 1
        frequencies = list(char_counts.values())
        max_original_frequency = 0
        if frequencies:
            max_original_frequency = max(frequencies)
        else:
            return 0
        max_length = 0
        for min_freq in range(max_original_frequency + 1):
            max_freq_allowed = min_freq + k
            current_length = 0
            for char in char_counts:
                original_freq = char_counts[char]
                if original_freq >= min_freq:
                    current_length += min(original_freq, max_freq_allowed)
            max_length = max(max_length, current_length)
        return len(word) - max_length