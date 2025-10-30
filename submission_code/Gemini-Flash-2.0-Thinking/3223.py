class Solution:
    def countCompleteSubstrings(self, word: str, k: int) -> int:
        n = len(word)
        count = 0
        for length in range(k, n + 1, k):
            for i in range(n - length + 1):
                substring = word[i : i + length]

                # Check condition 2: adjacent character difference
                is_adjacent_diff_ok = True
                for l in range(length - 1):
                    if abs(ord(substring[l + 1]) - ord(substring[l])) > 2:
                        is_adjacent_diff_ok = False
                        break

                if not is_adjacent_diff_ok:
                    continue

                # Check condition 1: character frequencies
                char_counts = {}
                for char in substring:
                    char_counts[char] = char_counts.get(char, 0) + 1

                is_frequency_ok = True
                if not char_counts:
                    continue

                if len(char_counts) * k != length:
                    continue

                for freq in char_counts.values():
                    if freq != k:
                        is_frequency_ok = False
                        break

                if is_frequency_ok:
                    count += 1
        return count