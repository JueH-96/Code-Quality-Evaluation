class Solution:
    def minimumOperations(self, num: str) -> int:
        n = len(num)
        options = []
        
        # Check all two-digit endings
        for ending in ["00", "25", "50", "75"]:
            d1, d2 = ending[0], ending[1]
            # Find the rightmost occurrence of d2
            last_d2 = None
            for i in range(n-1, -1, -1):
                if num[i] == d2:
                    last_d2 = i
                    break
            if last_d2 is None:
                continue
            # Find the rightmost occurrence of d1 before last_d2
            last_d1 = None
            for j in range(last_d2 - 1, -1, -1):
                if num[j] == d1:
                    last_d1 = j
                    break
            if last_d1 is None:
                continue
            # Calculate deletions
            deletions = (last_d2 - last_d1 - 1) + (n - last_d2 - 1)
            options.append(deletions)
        
        # Check for single '0' case
        if any(c == '0' for c in num):
            options.append(n - 1)
        
        # All-deletions case
        options.append(n)
        
        return min(options)