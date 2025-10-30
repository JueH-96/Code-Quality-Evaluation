class Solution:
    def countCompleteSubstrings(self, word: str, k: int) -> int:
        count = 0
        n = len(word)
        for i in range(n):
            for j in range(i, n):
                substring = word[i:j+1]
                if not substring:
                    continue

                char_counts = {}
                for char in substring:
                    char_counts[char] = char_counts.get(char, 0) + 1

                is_complete_freq = True
                if not char_counts:
                    is_complete_freq = False
                else:
                    required_chars = set(char_counts.keys())
                    for char in required_chars:
                        if char_counts[char] != k:
                            is_complete_freq = False
                            break
                    if is_complete_freq:
                        if len(required_chars) * k != len(substring):
                            is_complete_freq = False

                if not is_complete_freq:
                    continue

                is_complete_adj = True
                if len(substring) > 1:
                    for l in range(len(substring) - 1):
                        if abs(ord(substring[l]) - ord(substring[l+1])) > 2:
                            is_complete_adj = False
                            break

                if is_complete_freq and is_complete_adj:
                    count += 1
        return count