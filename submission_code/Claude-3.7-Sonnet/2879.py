class Solution:
    def minimumChanges(self, s: str, k: int) -> int:
        n = len(s)
        
        # Function to calculate minimum changes needed to make a substring a semi-palindrome
        def min_changes_to_semi_palindrome(start, end):
            substring = s[start:end+1]
            length = end - start + 1
            
            if length == 1:
                return float('inf')  # A semi-palindrome must have length at least 2
            
            min_changes = float('inf')
            
            # Try all valid values of d
            for d in range(1, length):
                if length % d == 0:
                    changes = 0
                    
                    # Group characters based on their indices modulo d
                    for i in range(d):
                        group = [substring[j] for j in range(i, length, d)]
                        
                        # Check if the group forms a palindrome
                        for j in range(len(group) // 2):
                            if group[j] != group[len(group) - 1 - j]:
                                changes += 1
                    
                    min_changes = min(min_changes, changes)
            
            return min_changes
        
        # Memoize results of min_changes_to_semi_palindrome
        cost_memo = {}
        def get_cost(start, end):
            if (start, end) not in cost_memo:
                cost_memo[(start, end)] = min_changes_to_semi_palindrome(start, end)
            return cost_memo[(start, end)]
        
        # Precompute costs for all possible substrings
        for i in range(n):
            for j in range(i+1, n):
                get_cost(i, j)
        
        # DP[i][j] = min changes to partition first i+1 characters into j substrings
        dp = [[float('inf')] * (k+1) for _ in range(n)]
        
        # Base case: first substring
        for i in range(1, n):
            dp[i][1] = get_cost(0, i)
        
        # Fill dp table
        for i in range(1, n):
            for j in range(2, min(i+1, k)+1):
                for prev in range(j-2, i):
                    if dp[prev][j-1] != float('inf'):
                        dp[i][j] = min(dp[i][j], dp[prev][j-1] + get_cost(prev+1, i))
        
        return dp[n-1][k]