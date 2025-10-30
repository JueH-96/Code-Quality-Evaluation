class Solution:
    def minimumCost(self, s: str) -> int:
        n = len(s)
        
        def getCost(target):
            cost = 0
            i = 0
            
            while i < n:
                if s[i] != target:
                    # Find the end of this segment that needs flipping
                    j = i
                    while j < n and s[j] != target:
                        j += 1
                    
                    # We need to flip segment [i, j-1]
                    # Option 1: flip from start (0 to j-1), cost = j
                    # Option 2: flip from end (i to n-1), cost = n - i
                    cost += min(j, n - i)
                    i = j
                else:
                    i += 1
            
            return cost
        
        # Try making all characters '0' or '1'
        cost0 = getCost('0')
        cost1 = getCost('1')
        
        return min(cost0, cost1)