from collections import defaultdict

class Solution:
    def maxPalindromesAfterOperations(self, words: List[str]) -> int:
        # Calculate total counts of each character
        total_counts = defaultdict(int)
        for word in words:
            for c in word:
                total_counts[c] += 1
        
        total_pairs = sum(cnt // 2 for cnt in total_counts.values())
        total_singles = sum(cnt % 2 for cnt in total_counts.values())
        
        # Initialize DP
        # dp[k] is a dictionary where key is sum_pairs and value is the minimal count_odds
        max_k = len(words)
        dp = [defaultdict(lambda: float('inf')) for _ in range(max_k + 1)]
        dp[0][0] = 0  # 0 words selected, sum_pairs 0, count_odds 0
        
        for word in words:
            l = len(word)
            pairs = l // 2
            is_odd = (l % 2 == 1)
            
            # Iterate backwards to prevent reusing the same word multiple times in the same step
            for current_k in reversed(range(max_k + 1)):
                if current_k > max_k:
                    continue
                # Iterate over a copy of the current states to avoid modifying the dictionary during iteration
                current_states = list(dp[current_k].items())
                for sum_p, current_odds in current_states:
                    new_k = current_k + 1
                    new_sum = sum_p + pairs
                    new_odds = current_odds + (1 if is_odd else 0)
                    
                    if new_sum > total_pairs:
                        continue
                    if new_k > max_k:
                        continue
                    
                    # Update the DP for the new state
                    if new_sum in dp[new_k]:
                        if new_odds < dp[new_k][new_sum]:
                            dp[new_k][new_sum] = new_odds
                    else:
                        dp[new_k][new_sum] = new_odds
        
        # Check for each possible k from max_k down to 0
        for k in range(max_k, 0, -1):
            if not dp[k]:
                continue
            for sum_p in dp[k]:
                if sum_p <= total_pairs:
                    odds = dp[k][sum_p]
                    if odds <= total_singles:
                        return k
        
        return 0