class Solution:
    def doesValidArrayExist(self, derived: List[int]) -> bool:
        # Let original[0] be an arbitrary binary value x.
        # Then, original[1] = x ⨁ derived[0], original[2] = original[1] ⨁ derived[1] = x ⨁ derived[0] ⨁ derived[1],
        # and so on.
        # After computing original[n-1], based on the cyclical definition, we have:
        # original[0] (which is x) must equal original[n-1] ⨁ derived[n-1].
        # That means:
        # x = (x ⨁ derived[0] ⨁ ... ⨁ derived[n-2]) ⨁ derived[n-1],
        # x = x ⨁ (derived[0] ⨁ ... ⨁ derived[n-1]).
        # Cancelling x on both sides (note that for any bit, a ⨁ a = 0) gives:
        # 0 = derived[0] ⨁ ... ⨁ derived[n-1].
        # Thus, a valid original exists if and only if the XOR of all elements in derived is 0.
        
        total_xor = 0
        for num in derived:
            total_xor ^= num
        return total_xor == 0