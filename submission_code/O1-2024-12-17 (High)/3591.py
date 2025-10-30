class Solution:
    def shiftDistance(self, s: str, t: str, nextCost: List[int], previousCost: List[int]) -> int:
        # Precompute the cost of shifting from any letter i to any letter j forwardly
        forwardDist = [[0]*26 for _ in range(26)]
        for i in range(26):
            cost = 0
            current = i
            # We'll make 25 forward steps (covering all possible different letters)
            for _ in range(1, 26):
                cost += nextCost[current]
                current = (current + 1) % 26
                forwardDist[i][current] = cost
        
        # Precompute the cost of shifting from any letter i to any letter j backwardly
        backwardDist = [[0]*26 for _ in range(26)]
        for i in range(26):
            cost = 0
            current = i
            # We'll make 25 backward steps
            for _ in range(1, 26):
                cost += previousCost[current]
                current = (current - 1) % 26
                backwardDist[i][current] = cost
        
        # Now compute the total minimal cost to transform s into t
        total_cost = 0
        for i in range(len(s)):
            if s[i] == t[i]:
                continue
            c1 = ord(s[i]) - ord('a')
            c2 = ord(t[i]) - ord('a')
            # The minimal cost to go from c1 to c2
            total_cost += min(forwardDist[c1][c2], backwardDist[c1][c2])
        
        return total_cost