class Solution:
    def lengthAfterTransformations(self, s: str, t: int) -> int:
        MOD = 10**9 + 7
        n = len(s)
        num_z = s.count('z')
        
        # Calculate the number of characters that will become 'z' within t-1 steps
        # and hence contribute to length increase at step t
        num_add = 0
        for c in s:
            if c != 'z':
                steps_to_z = ord('z') - ord(c)
                if steps_to_z + 1 <= t:
                    num_add += 1
                    
        # Total length is initial length plus contributions from 'z's and other characters
        total_length = (n + num_z + num_add) % MOD
        return total_length