import collections

class Solution:
    def makeStringGood(self, s: str) -> int:
        n = len(s)
        if n == 0:
            return 0
        counts = collections.Counter(s)
        
        # We test a set of candidate values for the target frequency k.
        # An optimal k is likely to be near n/m for some m in [1, 26].
        candidate_k = {1}
        for m in range(1, 27):
            if n // m > 0:
                candidate_k.add(n // m)
            candidate_k.add(n // m + 1)

        min_total_cost = n # Upper bound: delete all characters

        for k in candidate_k:
            if k == 0: continue
            
            # dp_prev[flow]: min cost for prefix processed before this char,
            # with `flow` items passed to the current character.
            dp_prev = [float('inf')] * (n + 1)
            dp_prev[0] = 0

            for char_code in range(26):
                char = chr(ord('a') + char_code)
                char_count = counts[char]
                
                # Precompute for O(N) DP transition
                h_minus_y = [(dp_prev[y] - y) if dp_prev[y] != float('inf') else float('inf') for y in range(n + 1)]
                h_plus_y = [(dp_prev[y] + y) if dp_prev[y] != float('inf') else float('inf') for y in range(n + 1)]

                pref_min_h_minus_y = [float('inf')] * (n + 1)
                current_min = float('inf')
                for y in range(n + 1):
                    current_min = min(current_min, h_minus_y[y])
                    pref_min_h_minus_y[y] = current_min
                
                suff_min_h_plus_y = [float('inf')] * (n + 2)
                current_min = float('inf')
                for y in range(n, -1, -1):
                    current_min = min(current_min, h_plus_y[y])
                    suff_min_h_plus_y[y] = current_min

                min_h_plus_y = suff_min_h_plus_y[0]

                dp_curr = [float('inf')] * (n + 1)
                for flow_out in range(n + 1):
                    # Option 1: Final count for this char is k
                    # Cost = flow_out + min_flow_in(dp_prev[flow_in] + |char_count + flow_in - flow_out - k|)
                    # Let A = flow_out - char_count + k. We need min_y(dp_prev[y] + |y - A|).
                    A = flow_out - char_count + k
                    
                    # term1 corresponds to min_{y<=A}(dp_prev[y] + A - y)
                    term1 = float('inf')
                    if A >= 0:
                        idx = min(A, n)
                        if pref_min_h_minus_y[idx] != float('inf'):
                            term1 = A + pref_min_h_minus_y[idx]
                    
                    # term2 corresponds to min_{y>A}(dp_prev[y] + y - A)
                    term2 = float('inf')
                    if A < n:
                        idx = max(A + 1, 0)
                        if suff_min_h_plus_y[idx] != float('inf'):
                             term2 = -A + suff_min_h_plus_y[idx]
                    
                    H_A = min(term1, term2)
                    cost1 = float('inf')
                    if H_A != float('inf'):
                       cost1 = flow_out + H_A

                    # Option 2: Final count for this char is 0
                    # Cost = min_flow_in(dp_prev[flow_in] + char_count + flow_in)
                    cost2 = float('inf')
                    if min_h_plus_y != float('inf'):
                        cost2 = char_count + min_h_plus_y

                    dp_curr[flow_out] = min(cost1, cost2)

                dp_prev = dp_curr

            # Final cost for this k is when flow out of 'z' is 0
            cost_for_k = dp_prev[0]
            min_total_cost = min(min_total_cost, cost_for_k)
            
        return min_total_cost