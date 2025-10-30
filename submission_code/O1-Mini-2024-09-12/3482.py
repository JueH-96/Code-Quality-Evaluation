from typing import List

class Solution:
    def minimumCost(self, target: str, words: List[str], costs: List[int]) -> int:
        from collections import defaultdict
        import sys

        n = len(target)
        dp = [sys.maxsize] * (n + 1)
        dp[0] = 0

        # Group words by their first character for efficient access
        word_dict = defaultdict(list)
        for word, cost in zip(words, costs):
            word_dict[word[0]].append((word, cost))

        for i in range(n):
            if dp[i] == sys.maxsize:
                continue
            first_char = target[i]
            for word, cost in word_dict.get(first_char, []):
                end = i + len(word)
                if end > n:
                    continue
                if target[i:end] == word:
                    if dp[end] > dp[i] + cost:
                        dp[end] = dp[i] + cost

        return dp[n] if dp[n] != sys.maxsize else -1