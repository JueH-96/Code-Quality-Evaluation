class Solution:
    def minimumCost(self, target: str, words: List[str], costs: List[int]) -> int:
        n = len(target)
        dp = [float('inf')] * (n + 1)
        dp[0] = 0  # Base case: empty string has cost 0
        
        # Preprocess each word to find all occurrences and their end positions
        end_words = [[] for _ in range(n + 1)]
        for i in range(len(words)):
            word = words[i]
            cost = costs[i]
            len_word = len(word)
            if len_word == 0 or len_word > n:
                continue
            s = 0
            while True:
                pos = target.find(word, s)
                if pos == -1:
                    break
                end = pos + len_word
                if end > n:
                    s = pos + 1
                    continue
                end_words[end].append((word, cost))
                s = pos + 1
        
        # Update dp based on the end positions
        for i in range(1, n + 1):
            for word, cost in end_words[i]:
                len_word = len(word)
                prev_i = i - len_word
                if prev_i >= 0:
                    if dp[prev_i] + cost < dp[i]:
                        dp[i] = dp[prev_i] + cost
        
        return dp[n] if dp[n] != float('inf') else -1