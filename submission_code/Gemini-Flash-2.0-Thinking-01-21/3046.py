class Solution:
    def minimumOperations(self, num: str) -> int:
        n = len(num)
        min_ops = n
        if '0' in num:
            min_ops = min(min_ops, n - 1)
        targets = ["00", "25", "50", "75"]
        for target in targets:
            d1, d2 = target[0], target[1]
            for j in range(n - 1, -1, -1):
                if num[j] == d2:
                    for i in range(j - 1, -1, -1):
                        if num[i] == d1:
                            deletions = max(0, j - i - 1) + max(0, n - 1 - j)
                            min_ops = min(min_ops, deletions)
        
        # Special case: if the target is "0"
        if '0' in num:
            min_ops = min(min_ops, n - 1)
        else:
            min_ops = min(min_ops, n)
            
        # Handle the case where the resulting number is just "0". 
        # This is already covered if '0' is in num, by taking n-1 operations. 
        # If no '0' in num, then we need n operations to get "0". 
        # Initial min_ops was set to n. So, this case is already considered. 

        return min_ops