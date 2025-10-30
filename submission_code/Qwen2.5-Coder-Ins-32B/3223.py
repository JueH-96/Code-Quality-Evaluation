class Solution:
    def countCompleteSubstrings(self, word: str, k: int) -> int:
        def is_valid(s, k):
            from collections import Counter
            count = Counter(s)
            return all(v == k for v in count.values())

        def count_for_unique_chars(unique_chars):
            n = len(word)
            count = 0
            for start in range(n):
                end = start
                while end < n and abs((ord(word[end]) - ord(word[end - 1]))) <= 2:
                    if len(set(word[start:end + 1])) > unique_chars:
                        break
                    if end - start + 1 == unique_chars * k and is_valid(word[start:end + 1], k):
                        count += 1
                    end += 1
            return count

        total_count = 0
        for unique_chars in range(1, 27):
            total_count += count_for_unique_chars(unique_chars)

        return total_count