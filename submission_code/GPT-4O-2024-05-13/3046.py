class Solution:
    def minimumOperations(self, num: str) -> int:
        n = len(num)
        min_operations = float('inf')
        
        # Check for all possible pairs that can form a number divisible by 25
        for i in range(n):
            for j in range(i + 1, n):
                if (num[i] == '2' and num[j] == '5') or (num[i] == '5' and num[j] == '0') or (num[i] == '7' and num[j] == '5') or (num[i] == '0' and num[j] == '0'):
                    # Calculate the number of deletions needed
                    deletions = (n - 1 - j) + (j - 1 - i)
                    min_operations = min(min_operations, deletions)
        
        # If no valid pair is found, we need to delete all digits except one '0'
        if min_operations == float('inf'):
            min_operations = n - 1
        
        return min_operations