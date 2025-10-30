class Solution:
    def minimumDistance(self, points: List[List[int]]) -> int:
        import sys
        from math import inf
        
        n = len(points)
        # Precompute W = x+y and Z = x-y for each point
        W = []
        Z = []
        for i, (x, y) in enumerate(points):
            W.append((x + y, i))
            Z.append((x - y, i))
        
        # Sort W and Z by their transformed values
        W.sort(key=lambda p: p[0])
        Z.sort(key=lambda p: p[0])
        
        # Extract top 2 and bottom 2 for W
        Wmin1, Wmin1Idx = W[0][0],  W[0][1]   # smallest
        Wmin2, Wmin2Idx = W[1][0],  W[1][1]   # 2nd smallest
        Wmax1, Wmax1Idx = W[-1][0], W[-1][1]  # largest
        Wmax2, Wmax2Idx = W[-2][0], W[-2][1]  # 2nd largest
        
        # Extract top 2 and bottom 2 for Z
        Zmin1, Zmin1Idx = Z[0][0],  Z[0][1]
        Zmin2, Zmin2Idx = Z[1][0],  Z[1][1]
        Zmax1, Zmax1Idx = Z[-1][0], Z[-1][1]
        Zmax2, Zmax2Idx = Z[-2][0], Z[-2][1]
        
        min_dist = sys.maxsize
        
        # Check each point if removed
        for i in range(n):
            # Compute new Wmax if i is removed
            if i == Wmax1Idx:
                newWmax = Wmax2
            else:
                newWmax = Wmax1
            
            # Compute new Wmin if i is removed
            if i == Wmin1Idx:
                newWmin = Wmin2
            else:
                newWmin = Wmin1
            
            # Compute new Zmax if i is removed
            if i == Zmax1Idx:
                newZmax = Zmax2
            else:
                newZmax = Zmax1
            
            # Compute new Zmin if i is removed
            if i == Zmin1Idx:
                newZmin = Zmin2
            else:
                newZmin = Zmin1
            
            # Manhattan distance is max(|Wmax-Wmin|, |Zmax-Zmin|)
            # but we've stored them as integers, so we can do direct subtraction:
            current_max_dist = max(newWmax - newWmin, newZmax - newZmin)
            min_dist = min(min_dist, current_max_dist)
        
        return min_dist