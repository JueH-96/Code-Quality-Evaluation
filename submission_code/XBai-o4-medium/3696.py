import math

class Solution:
    def countSubstrings(self, s: str) -> int:
        # Precompute m_d for each digit 1-9
        m = [0] * 10  # m[0] is unused
        for d in range(1, 10):
            g = math.gcd(10, d)
            m[d] = d // g
        
        # Initialize current_mods for each digit 1-9
        current_mods = [None] * 10  # current_mods[d] holds the counts for digit d
        for d in range(1, 10):
            md = m[d]
            current_mods[d] = [0] * md
        
        ans = 0
        
        for c in s:
            curr = int(c)
            # Create new_mods for each digit
            new_mods = [None] * 10
            for d in range(1, 10):
                md = m[d]
                # Create a new list for new remainders
                new_m = [0] * md
                prev = current_mods[d]
                # Process each previous remainder
                for r in range(md):
                    cnt = prev[r]
                    if cnt > 0:
                        new_r = (r * 10 + curr) % md
                        new_m[new_r] += cnt
                # Add the new substring which is just the current digit
                new_curr_mod = curr % md
                new_m[new_curr_mod] += 1
                new_mods[d] = new_m
            # Update current_mods with new_mods
            for d in range(1, 10):
                current_mods[d] = new_mods[d]
            # If current digit is non-zero, add the count of valid substrings ending here
            if curr != 0:
                ans += current_mods[curr][0]
        
        return ans