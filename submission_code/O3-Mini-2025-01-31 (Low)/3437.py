from typing import List
import bisect

class Solution:
    def maximumTotalDamage(self, power: List[int]) -> int:
        # First compute the total damage for each distinct spell damage
        damage_map = {}
        for p in power:
            if p in damage_map:
                damage_map[p] += p
            else:
                damage_map[p] = p
        
        # Create a sorted list of unique damages
        keys = sorted(damage_map.keys())
        n = len(keys)
        
        # Prepare a dp array where dp[i] is the maximum damage we can get using keys[0...i]
        dp = [0] * n
        
        # For convenience, store the values corresponding to each key
        values = [damage_map[k] for k in keys]
        
        dp[0] = values[0]
        
        for i in range(1, n):
            # Option 1: skip current key: damage = dp[i-1]
            option1 = dp[i-1]
            
            # Option 2: take current key: damage = value of current key + dp[index] where keys[index] <= keys[i]-3
            # We need to find the rightmost index j such that keys[j] <= keys[i]-3.
            current_key = keys[i]
            # bisect_right returns insertion point for (current_key-3)
            j = bisect.bisect_right(keys, current_key - 3) - 1
            option2 = values[i]
            if j >= 0:
                option2 += dp[j]
            
            dp[i] = max(option1, option2)
        
        return dp[-1]