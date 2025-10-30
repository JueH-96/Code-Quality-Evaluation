class Solution:
    def minimumDistance(self, points: List[List[int]]) -> int:
        import sys
        
        n = len(points)
        # Precompute T1 = x + y and T2 = x - y for each point
        T1vals = [(points[i][0] + points[i][1], i) for i in range(n)]
        T2vals = [(points[i][0] - points[i][1], i) for i in range(n)]
        
        # Sort each list by the computed values
        T1vals.sort(key=lambda x: x[0])  # Sort by (x+y)
        T2vals.sort(key=lambda x: x[0])  # Sort by (x-y)
        
        # We only need to consider removing points that could affect
        # the extremes of T1 or T2. These are among the top two or bottom
        # two points in T1vals and T2vals. Gather up to 8 indices.
        candidates = set()
        candidates.add(T1vals[0][1])
        candidates.add(T1vals[1][1])
        candidates.add(T1vals[-1][1])
        candidates.add(T1vals[-2][1])
        candidates.add(T2vals[0][1])
        candidates.add(T2vals[1][1])
        candidates.add(T2vals[-1][1])
        candidates.add(T2vals[-2][1])
        
        # We'll compute the new extremes after removing each candidate
        # and track the minimum possible maximum distance.
        ans = float('inf')
        
        for idx in candidates:
            # Compute new minT1 and maxT1 if we remove the point idx
            if T1vals[0][1] != idx:
                newMinT1 = T1vals[0][0]
            else:
                newMinT1 = T1vals[1][0]
                
            if T1vals[-1][1] != idx:
                newMaxT1 = T1vals[-1][0]
            else:
                newMaxT1 = T1vals[-2][0]
            
            # Compute new minT2 and maxT2 if we remove the point idx
            if T2vals[0][1] != idx:
                newMinT2 = T2vals[0][0]
            else:
                newMinT2 = T2vals[1][0]
                
            if T2vals[-1][1] != idx:
                newMaxT2 = T2vals[-1][0]
            else:
                newMaxT2 = T2vals[-2][0]
            
            # The maximum Manhattan distance among the remaining points
            # is the maximum of (newMaxT1 - newMinT1) and (newMaxT2 - newMinT2).
            currMaxDist = max(newMaxT1 - newMinT1, newMaxT2 - newMinT2)
            ans = min(ans, currMaxDist)
        
        return ans