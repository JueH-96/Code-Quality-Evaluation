class Solution:
    def removeAlmostEqualCharacters(self, word: str) -> int:
        n = len(word)
        dp = [{} for _ in range(n + 1)]
        dp[0][None] = 0  # Starting with no previous character
        
        for i in range(n):
            for prev, cost in dp[i].items():
                for c in 'abcdefghijklmnopqrstuvwxyz':
                    if prev is not None:
                        if c == prev or abs(ord(c) - ord(prev)) == 1:
                            continue
                    new_cost = cost if c == word[i] else cost + 1
                    if c not in dp[i + 1] or new_cost < dp[i + 1][c]:
                        dp[i + 1][c] = new_cost
                        
        return min(dp[n].values())