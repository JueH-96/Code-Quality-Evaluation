class Solution:
    def findMinimumTime(self, strength: List[int], K: int) -> int:
        n = len(strength)
        ans = float('inf')
        
        for i in range(1 << n):
            perm = []
            for j in range(n):
                if (i >> j) & 1:
                    perm.append(j)
            
            if len(perm) != n:
                continue

            time = 0
            energy = 0
            x = 1
            possible = True
            
            for idx in perm:
                strength_needed = strength[idx]
                
                t = 0
                while energy < strength_needed:
                    energy += x
                    t += 1
                
                time += t
                energy = 0
                x += K
            
            ans = min(ans, time)
        return ans