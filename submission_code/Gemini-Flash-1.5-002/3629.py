class Solution:
    def lengthAfterTransformations(self, s: str, t: int) -> int:
        mod = 10**9 + 7
        length = len(s)
        z_count = s.count('z')
        
        for _ in range(t):
            new_length = length
            new_z_count = z_count
            z_count = new_length - z_count
            length = (new_length - new_z_count) + new_z_count * 2
            
            if length > mod:
                length %= mod

        return length