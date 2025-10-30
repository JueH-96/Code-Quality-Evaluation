from collections import defaultdict

class Solution:
    def minimumSubstringsInPartition(self, s: str) -> int:
        n = len(s)
        if n == 0:
            return 0
        
        # Precompute prefix counts
        prefix_counts = [defaultdict(int)]
        current = defaultdict(int)
        for c in s:
            current[c] += 1
            prefix_counts.append(dict(current))
        
        # Initialize DP array
        dp = [float('inf')] * n
        dp[-1] = 0  # Base case: empty substring after last character
        
        for i in range(n):
            for j in range(-1, i):
                if j == -1:
                    # Substring is from 0 to i
                    temp = prefix_counts[i+1]
                else:
                    # Substring is from j+1 to i
                    temp = {}
                    for c in prefix_counts[i+1]:
                        temp[c] = prefix_counts[i+1][c] - prefix_counts[j+1].get(c, 0)
                
                # Check if all characters in temp have the same count and non-zero
                if not temp:
                    continue  # Shouldn't happen as j < i
                first = next(iter(temp.values()))
                all_equal = True
                for v in temp.values():
                    if v != first:
                        all_equal = False
                        break
                if all_equal:
                    if first == 0:
                        continue  # All zeros, but substring is non-empty
                    if dp[j] + 1 < dp[i]:
                        dp[i] = dp[j] + 1
        
        return dp[-1]