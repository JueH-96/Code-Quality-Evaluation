class Solution:
    def countKeyChanges(self, s: str) -> int:
        if not s:
            return 0
        
        changes = 0
        # Track the key of the previous character in lowercase form
        prev_key = s[0].lower()
        
        for ch in s[1:]:
            # Determine the current key by lowercasing
            curr_key = ch.lower()
            # If the key differs from the previous, it's a key change
            if curr_key != prev_key:
                changes += 1
                prev_key = curr_key
            # Otherwise, keep the same prev_key
        
        return changes