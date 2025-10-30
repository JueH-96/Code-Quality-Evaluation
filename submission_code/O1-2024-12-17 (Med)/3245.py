class Solution:
    def beautifulIndices(self, s: str, a: str, b: str, k: int) -> List[int]:
        # Find all start indices where 'a' occurs in 's'
        positionsA = []
        lenA = len(a)
        for i in range(len(s) - lenA + 1):
            if s[i:i+lenA] == a:
                positionsA.append(i)
                
        # Find all start indices where 'b' occurs in 's'
        positionsB = []
        lenB = len(b)
        for i in range(len(s) - lenB + 1):
            if s[i:i+lenB] == b:
                positionsB.append(i)
        
        # Two-pointer approach to find all i in positionsA that have
        # some j in positionsB with |i-j| <= k
        result = []
        pB = 0
        for i in positionsA:
            # Move pB so that positionsB[pB] >= i - k
            while pB < len(positionsB) and positionsB[pB] < i - k:
                pB += 1
            
            # Now check if there's a j = positionsB[pB] within [i - k, i + k]
            if pB < len(positionsB) and positionsB[pB] <= i + k:
                result.append(i)
        
        return result