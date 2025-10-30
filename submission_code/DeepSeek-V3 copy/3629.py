class Solution:
    def lengthAfterTransformations(self, s: str, t: int) -> int:
        MOD = 10**9 + 7
        
        # Precompute the length changes for each character after t transformations
        # We need to find the length of the string after t transformations
        # For each character, we can determine how many times it will be transformed into 'z' and then into "ab"
        
        # For each character, we can simulate the transformations up to t steps
        # However, since t can be up to 1e5, we need a smarter approach
        
        # Let's consider the transformation rules:
        # For any character c, after one transformation:
        # if c == 'z': it becomes "ab" (length 2)
        # else: it becomes next character (length 1)
        
        # So, for each character, we can determine how many times it will be transformed into 'z' and then into "ab"
        
        # Let's precompute for each character the number of steps it takes to reach 'z'
        # For example, 'a' takes 25 steps to reach 'z' (a->b->c->...->z)
        # 'b' takes 24 steps, ..., 'y' takes 1 step, 'z' takes 0 steps
        
        # So, for each character c, the number of steps to reach 'z' is (ord('z') - ord(c)) % 26
        
        # Then, for each character, we can determine how many times it will be transformed into 'z' in t steps
        
        # Let's define for each character c:
        # steps_to_z = (ord('z') - ord(c)) % 26
        # if steps_to_z == 0 and c == 'z':
        #   then it will be transformed into "ab" every step
        
        # So, for each character, we can calculate the number of times it will be transformed into 'z' in t steps
        
        # Let's define for each character c:
        # if steps_to_z == 0:
        #   if c == 'z':
        #       then it will be transformed into "ab" every step
        #       so the length contribution is 2^t
        #   else:
        #       it will be transformed into the next character, but never into 'z' in t steps
        #       so the length contribution is 1
        # else:
        #   if t >= steps_to_z:
        #       then it will be transformed into 'z' at least once
        #       and then it will be transformed into "ab" every step after that
        #       so the length contribution is 2^(t - steps_to_z)
        #   else:
        #       it will not be transformed into 'z' in t steps
        #       so the length contribution is 1
        
        # Now, we can compute the total length by summing the contributions of each character
        
        total_length = 0
        for c in s:
            steps_to_z = (ord('z') - ord(c)) % 26
            if steps_to_z == 0:
                if c == 'z':
                    # It will be transformed into "ab" every step
                    # So the length is 2^t
                    length = pow(2, t, MOD)
                else:
                    # It will be transformed into the next character, but never into 'z' in t steps
                    length = 1
            else:
                if t >= steps_to_z:
                    # It will be transformed into 'z' at least once
                    # Then it will be transformed into "ab" every step after that
                    # So the length is 2^(t - steps_to_z)
                    length = pow(2, t - steps_to_z, MOD)
                else:
                    # It will not be transformed into 'z' in t steps
                    length = 1
            total_length = (total_length + length) % MOD
        return total_length