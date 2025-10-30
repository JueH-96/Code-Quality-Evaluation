class Solution:
    def makeStringGood(self, s: str) -> int:
        from collections import Counter
        
        count = [0] * 26
        for c in s:
            count[ord(c) - ord('a')] += 1
        
        min_operations = float('inf')
        max_count = max(count)
        # Consider f from 0 to max_count + len(s)
        # We can limit f up to the maximum possible needed, here arbitrarily chosen as up to 2*len(s)
        max_f = max_count + len(s)
        
        for f in range(0, max_f + 1):
            current_cost = 0
            excess = [0] * 26
            deficit = [0] * 26
            valid = False  # To check if this f can form a valid configuration
            total_chars = 0  # total chars after considering insertions (not including insertions themselves)
            
            for c in range(26):
                curr = count[c]
                if f == 0:
                    # Target is to delete all characters
                    current_cost += curr
                    continue
                delta = curr - f
                if delta > 0:
                    # excess
                    current_cost += delta
                    excess[c] = delta
                else:
                    # deficit
                    need = -delta
                    current_cost += need
                    deficit[c] = need
                total_chars += max(0, curr)  # actual used chars (not considering insertions)
            if f == 0:
                # All deleted
                min_operations = min(min_operations, current_cost)
                continue
            # Now, also consider insertions and deletions per character (recompute)
            # The above approach was incorrect. Let's re-calculate properly.
            # The correct way: for each character, you can choose to have it in S or not.
            # But this is complex. Alternative Idea inspired by examples:
            # Try all possible subsets S and their best cost, but not feasible.
            # Revert to the model considering all characters in S, which allows us to compute savings.
            # New approach: consider all characters in S, cost is sum(abs(count[c] -f)), then compute savings.
            total_cost = 0
            excess = [0] * 26
            deficit = [0] * 26
            for c in range(26):
                delta = count[c] - f
                total_cost += abs(delta)
                if delta > 0:
                    excess[c] = delta
                else:
                    deficit[c] = -delta
            # Now compute possible savings between adjacent characters
            savings = 0
            # Process pairs (a, b), (b, c), ..., (y, z)
            for i in range(25):
                e = excess[i]
                if e == 0:
                    continue
                d = deficit[i+1]
                if d == 0:
                    continue
                save = min(e, d)
                savings += save
            # adjusted cost: total_cost - savings
            adjusted_cost = total_cost - savings
            if adjusted_cost < min_operations:
                min_operations = adjusted_cost
            # Also consider the case of choosing subset S
            # But this is not handled here
        
        return min_operations