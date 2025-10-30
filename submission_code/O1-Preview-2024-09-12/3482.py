class Solution:
    def minimumCost(self, target: str, words: List[str], costs: List[int]) -> int:
        from collections import defaultdict
        word_costs = {}
        for word, cost in zip(words, costs):
            if word not in word_costs or cost < word_costs[word]:
                word_costs[word] = cost
        word_lengths = set(len(word) for word in word_costs)
        n = len(target)
        dp = [float('inf')] * (n + 1)
        dp[0] = 0
        for i in range(1, n + 1):
            for len_word in word_lengths:
                if i >= len_word:
                    subword = target[i - len_word:i]
                    if subword in word_costs:
                        dp[i] = min(dp[i], dp[i - len_word] + word_costs[subword])
        return dp[n] if dp[n] != float('inf') else -1