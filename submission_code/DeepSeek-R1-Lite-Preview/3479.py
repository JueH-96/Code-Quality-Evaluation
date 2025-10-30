import math

class Solution:
    def numberOfSubstrings(self, s: str) -> int:
        n = len(s)
        # Compute prefix sums of zeros
        prefix_z = [0] * (n + 1)
        for i in range(n):
            prefix_z[i+1] = prefix_z[i] + (1 if s[i] == '0' else 0)
        
        # Compute z_max_total
        z_max_total = int( (-1 + math.isqrt(1 + 4*n)) // 2 )
        
        answer = 0
        for z in range(0, z_max_total + 1):
            L_min = z * z + z
            if L_min > n:
                continue
            # Initialize frequency map
            freq = {}
            i = 0
            for j in range(L_min, n + 1):
                # Move i to the start of the window
                while i < j - L_min + 1:
                    pz = prefix_z[i]
                    freq[pz] = freq.get(pz, 0) + 1
                    i += 1
                # Current target is prefix_z[j] - z
                target = prefix_z[j] - z
                answer += freq.get(target, 0)
        
        return answer