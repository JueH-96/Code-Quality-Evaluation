class Solution:
    def minCostGoodCaption(self, caption: str) -> str:
        n = len(caption)
        if n < 3:
            return ""
        
        # Precompute prefix sums for each letter 'a' to 'z'
        # pref[d][i] = total cost to convert caption[0:i] to letter d.
        # cost for index i (0-indexed) is abs(ord(caption[i]) - ord(d))
        pref = {d: [0]*(n+1) for d in "abcdefghijklmnopqrstuvwxyz"}
        for d in "abcdefghijklmnopqrstuvwxyz":
            for i in range(n):
                cost_here = abs(ord(caption[i]) - ord(d))
                pref[d][i+1] = pref[d][i] + cost_here
        
        # dp[i] = (min_cost, transformed_string) for caption[i:]
        # Use None for unreachable
        dp_cost = [float('inf')] * (n+1)
        dp_str = [""] * (n+1)
        
        dp_cost[n] = 0
        dp_str[n] = ""
        
        # For lexicographical comparisons we compare tuples (cost, string)
        # Process from rightmost index backwards.
        # We'll try segments with length L>=3.
        for i in range(n-1, -1, -1):
            best_cost = float('inf')
            best_res = None
            # We must have at least 3 chars in a group.
            for j in range(i+2, n):  # group from i to j; length = j-i+1
                # Determine the best letter d for segment [i, j]
                seg_cost = float('inf')
                best_letter = None
                # For candidate letter d in alphabetical order (so that if tie we pick lex smallest)
                for d in "abcdefghijklmnopqrstuvwxyz":
                    cost_seg = pref[d][j+1] - pref[d][i]
                    # update if lower cost
                    if cost_seg < seg_cost:
                        seg_cost = cost_seg
                        best_letter = d
                total_cost = seg_cost + dp_cost[j+1]
                if total_cost < best_cost:
                    best_cost = total_cost
                    best_res = best_letter * (j - i + 1) + dp_str[j+1]
                elif total_cost == best_cost:
                    candidate = best_letter * (j - i + 1) + dp_str[j+1]
                    if candidate < best_res:
                        best_res = candidate
            dp_cost[i] = best_cost
            dp_str[i] = best_res if best_res is not None else None
        
        # If dp_cost[0] is inf then impossible; otherwise return the string.
        return dp_str[0] if dp_cost[0] != float('inf') else ""