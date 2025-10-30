class Solution:
    def minimumOperations(self, num: str) -> int:
        n = len(num)
        positions = {'0': [], '2': [], '5': [], '7': []}
        
        # Populate positions for each digit of interest
        for i in range(n):
            if num[i] in positions:
                positions[num[i]].append(i)
        
        min_deletions = n  # Deleting all digits to get 0
        
        # Check for pairs forming '00'
        for i in positions['0']:
            for j in positions['0']:
                if j > i:
                    deletions = n - (i + 2)
                    min_deletions = min(min_deletions, deletions)
        
        # Check for pairs forming '25'
        for i in positions['2']:
            for j in positions['5']:
                if j > i:
                    deletions = n - (i + 2)
                    min_deletions = min(min_deletions, deletions)
        
        # Check for pairs forming '50'
        for i in positions['5']:
            for j in positions['0']:
                if j > i:
                    deletions = n - (i + 2)
                    min_deletions = min(min_deletions, deletions)
        
        # Check for pairs forming '75'
        for i in positions['7']:
            for j in positions['5']:
                if j > i:
                    deletions = n - (i + 2)
                    min_deletions = min(min_deletions, deletions)
        
        # Check if there is at least one '0' to delete all except one '0'
        if positions['0']:
            min_deletions = min(min_deletions, n - 1)
        
        return min_deletions