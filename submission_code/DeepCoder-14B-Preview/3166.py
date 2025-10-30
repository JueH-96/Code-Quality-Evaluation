from collections import Counter

class Solution:
    def minGroupsForValidAssignment(self, nums: list[int]) -> int:
        freq = Counter(nums)
        n = len(nums)
        
        for G in range(1, n + 1):
            s = n // G
            sum_g_min = 0
            sum_g_max = 0
            valid = True
            
            for x in freq:
                c = freq[x]
                # Calculate minimal and maximal possible groups for this x
                g_min = (c + s) // (s + 1) if s != 0 else 0
                g_max = c // s if s != 0 else (0 if c == 0 else float('inf'))
                
                if g_min > g_max:
                    valid = False
                    break
                sum_g_min += g_min
                sum_g_max += g_max
            
            if not valid:
                continue
            
            # Check if the total groups can be achieved with the current G
            if sum_g_min <= G <= sum_g_max:
                return G
        
        # If no valid G found (which theoretically shouldn't happen as G=n is always valid)
        return n