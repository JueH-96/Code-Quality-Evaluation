class Solution:
    def minimumCost(self, s: str) -> int:
        def solve(target):
            n = len(s)
            cost = 0
            current = [int(c) for c in s]
            
            for i in range(n):
                if current[i] != target:
                    cost1 = i + 1  # cost of operation 1 with index i
                    cost2 = n - i  # cost of operation 2 with index i
                    
                    if cost1 <= cost2:
                        # Use operation 1: flip characters 0 to i
                        for j in range(i + 1):
                            current[j] = 1 - current[j]
                        cost += cost1
                    else:
                        # Use operation 2: flip characters i to n-1
                        for j in range(i, n):
                            current[j] = 1 - current[j]
                        cost += cost2
            
            return cost
        
        # Try both targets: all 0s and all 1s
        return min(solve(0), solve(1))