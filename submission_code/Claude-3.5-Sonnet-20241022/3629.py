class Solution:
    def lengthAfterTransformations(self, s: str, t: int) -> int:
        MOD = 10**9 + 7
        
        # Count the number of z's and non-z's that will appear after i transformations
        def count_after_i_transformations(i: int, curr_z: int, curr_nonz: int) -> tuple:
            if i == 0:
                return curr_z, curr_nonz
            
            # All non-z characters except 'y' will remain non-z
            # All 'y' characters will become 'z'
            # All 'z' characters will become two non-z characters ('ab')
            new_nonz = (curr_nonz - curr_z + 2 * curr_z) % MOD  # non-z + 2*z
            new_z = curr_nonz % MOD  # all non-z become z in next transformation
            
            return count_after_i_transformations(i - 1, new_z, new_nonz)
        
        # Count initial z's and non-z's in the string
        initial_z = s.count('z')
        initial_nonz = len(s) - initial_z
        
        # Get final counts after t transformations
        final_z, final_nonz = count_after_i_transformations(t, initial_z, initial_nonz)
        
        # Final length is: z's (each contributes 2 to length) + non-z's (each contributes 1)
        return (2 * final_z + final_nonz) % MOD