class Solution:
    def minimumCost(self, s: str) -> int:
        n = len(s)
        
        # `current_state` represents the effective value of the characters that we are currently
        # trying to make uniform. It starts with the value of the first character s[0].
        # If s[0] is '0', current_state is 0. If s[0] is '1', current_state is 1.
        current_state = int(s[0])
        
        total_cost = 0
        
        # We iterate from the second character (index 1) to the end of the string.
        # For each character s[i], we compare its original value to `current_state`.
        # `current_state` essentially reflects the effective value of s[i-1] after any
        # necessary operations have been applied to make s[0]...s[i-1] uniform.
        for i in range(1, n):
            # `char_val_at_i` is the original integer value of s[i].
            char_val_at_i = int(s[i])
            
            # If the original s[i] is different from our `current_state`,
            # it means there's a "transition" or "mismatch" between the effective value
            # of s[i-1] (represented by `current_state`) and the original value of s[i].
            # To make s[i-1] and s[i] uniform, we must apply an operation.
            if char_val_at_i != current_state:
                # This mismatch is at the boundary between index (i-1) and index i.
                # To resolve this specific mismatch, we have two options:
                # 1. Apply operation type 1 at index (i-1): `invert(0, i-1)`.
                #    This operation flips s[0]...s[i-1]. Its cost is (i-1) + 1 = i.
                #    Effectively, it changes s[i-1] to match s[i] (original value).
                # 2. Apply operation type 2 at index i: `invert(i, n-1)`.
                #    This operation flips s[i]...s[n-1]. Its cost is n - i.
                #    Effectively, it changes s[i] to match s[i-1]'s current effective value.
                
                # We add the minimum cost of these two options to the total cost.
                total_cost += min(i, n - i)
                
                # After resolving the mismatch (by either flipping the prefix or suffix),
                # the characters s[i-1] and s[i] are now consistent.
                # However, this action effectively "flips" the expectation for the subsequent characters.
                # If s[i-1] was effectively '0' and s[i] was '1' (original), and we fixed it,
                # then for s[i+1], the new 'current_state' that it should match will be `1 - current_state`.
                # This is because we crossed a conceptual boundary where the prevailing desired character value flipped.
                current_state = 1 - current_state
                
        return total_cost