class Solution:
    def shiftDistance(self, s: str, t: str, nextCost: List[int], previousCost: List[int]) -> int:
        n = len(s)
        
        # Build arrays of length 52 repeating nextCost and previousCost
        # so we can easily handle wrapping via prefix sums.
        nextArr = [0] * 52
        prevArr = [0] * 52
        for i in range(52):
            nextArr[i] = nextCost[i % 26]
            prevArr[i] = previousCost[i % 26]
        
        # Build prefix sums for each array.
        nextPrefix = [0] * 53  # nextPrefix[i+1] = sum of nextArr[0..i]
        prevPrefix = [0] * 53
        for i in range(52):
            nextPrefix[i + 1] = nextPrefix[i] + nextArr[i]
            prevPrefix[i + 1] = prevPrefix[i] + prevArr[i]
        
        total_cost = 0
        
        for i in range(n):
            cIndex = ord(s[i]) - ord('a')
            dIndex = ord(t[i]) - ord('a')
            
            if cIndex == dIndex:
                # No cost if already the same character
                continue
            
            # Forward distance and cost
            forwardDist = (dIndex - cIndex) % 26
            forwardCost = nextPrefix[cIndex + forwardDist] - nextPrefix[cIndex]
            
            # Backward distance and cost
            backwardDist = (cIndex - dIndex) % 26
            # We shift the start index by +26 to avoid negative indices in prevArr
            start = cIndex + 26
            suba = start - (backwardDist - 1)  # inclusive
            subb = start                       # inclusive
            backwardCost = prevPrefix[subb + 1] - prevPrefix[suba]
            
            # Take the minimum of forward or backward shifting
            total_cost += min(forwardCost, backwardCost)
        
        return total_cost