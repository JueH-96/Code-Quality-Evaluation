class Solution:
    def minOperations(self, s1: str, s2: str, x: int) -> int:
        n = len(s1)
        # Quick check: if the total number of mismatched bits is odd, it's impossible
        diff_positions = [i for i in range(n) if s1[i] != s2[i]]
        if len(diff_positions) % 2 == 1:
            return -1
        
        cost = 0
        pending = False  # Tracks if we have one unmatched mismatch we're holding to pair
        
        i = 0
        while i < n:
            if s1[i] == s2[i]:
                i += 1
                continue
            
            # We have a mismatch at position i
            if pending:
                # We already had one unmatched mismatch, pair with this mismatch
                cost += min(x, 2)
                pending = False
                i += 1
            else:
                # Check for a "cross-mismatch" with the next position
                if (
                    i + 1 < n
                    and s1[i] != s2[i]      # mismatch at i
                    and s1[i+1] != s2[i+1] # mismatch at i+1
                    and s1[i] == s2[i+1]   # crossing pattern
                    and s1[i+1] == s2[i]
                ):
                    # We can fix both mismatches at once with a single operation of cost=1
                    # (flip consecutive i, i+1), or with operation of cost=x (flip any two bits).
                    # The cheaper approach is cost = min(x, 1). But x>=1, so actually cost = 1 if x>1.
                    cost += min(x, 1)
                    i += 2
                else:
                    # No cross mismatch with the next bit, so leave this mismatch pending.
                    pending = True
                    i += 1
        
        # If there's still one mismatch pending, we cannot fix it alone (needs pairs)
        if pending:
            return -1
        
        return cost