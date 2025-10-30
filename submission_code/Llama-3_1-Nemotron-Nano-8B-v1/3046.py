class Solution:
    def minimumOperations(self, num: str) -> int:
        ans = len(num)  # Initialize with the maximum possible deletions (all digits)
        n = len(num)
        
        # Check all possible pairs (i, j) where i < j
        for i in range(n):
            for j in range(i + 1, n):
                pair = (num[i], num[j])
                if pair in {('0', '0'), ('2', '5'), ('5', '0'), ('7', '5')}:
                    deletions = n - i - 2
                    if deletions < ans:
                        ans = deletions
        
        # Check if there's a zero to consider the case of leaving a single zero
        if '0' in num:
            candidate = n - 1
            if candidate < ans:
                ans = candidate
        
        return ans