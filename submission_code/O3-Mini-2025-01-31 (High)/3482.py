from typing import List
import math

class Solution:
    def minimumCost(self, target: str, words: List[str], costs: List[int]) -> int:
        # Consolidate duplicate words by keeping the minimum cost for each unique word.
        word_to_cost = {}
        for w, c in zip(words, costs):
            if w in word_to_cost:
                word_to_cost[w] = min(word_to_cost[w], c)
            else:
                word_to_cost[w] = c
        
        # Build a mapping from the first character of the word to a list of (word, cost) pairs.
        # This allows us to only try the candidates that can possibly match at the current position.
        candidates = {}
        for w, c in word_to_cost.items():
            first = w[0]
            if first not in candidates:
                candidates[first] = []
            candidates[first].append((w, c))
        
        n = len(target)
        # dp[i] will store the minimum cost to form target[:i]
        dp = [math.inf] * (n + 1)
        dp[0] = 0
        
        # Iterate over each position in the target string.
        for i in range(n):
            # If the current prefix of target cannot be formed, skip.
            if dp[i] == math.inf:
                continue
            # Check only those words that start with the current character.
            ch = target[i]
            if ch not in candidates:
                continue
            for word, cost in candidates[ch]:
                l = len(word)
                # If the word fits in the remainder of target and matches exactly, update dp.
                if i + l <= n and target[i:i+l] == word:
                    dp[i+l] = min(dp[i+l], dp[i] + cost)
        
        return dp[n] if dp[n] != math.inf else -1

# Sample test cases
if __name__ == '__main__':
    sol = Solution()
    
    # Example 1:
    target = "abcdef"
    words = ["abdef", "abc", "d", "def", "ef"]
    costs = [100, 1, 1, 10, 5]
    print(sol.minimumCost(target, words, costs))  # Expected Output: 7

    # Example 2:
    target = "aaaa"
    words = ["z", "zz", "zzz"]
    costs = [1, 10, 100]
    print(sol.minimumCost(target, words, costs))  # Expected Output: -1