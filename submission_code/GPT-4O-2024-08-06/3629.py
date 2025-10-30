class Solution:
    def lengthAfterTransformations(self, s: str, t: int) -> int:
        MOD = 10**9 + 7
        
        # Initial length of the string
        length = len(s)
        
        # Count the number of 'z's in the initial string
        count_z = s.count('z')
        
        # For each transformation
        for _ in range(t):
            # Each 'z' becomes "ab", so it increases the length by 1 for each 'z'
            length = (length + count_z) % MOD
            
            # After transformation, every 'z' becomes "ab", so the number of 'z's becomes 0
            # Every 'y' becomes 'z', so we need to count new 'z's
            count_z = length - len(s)  # New 'z's are those that were 'y' in the previous step
        
        return length