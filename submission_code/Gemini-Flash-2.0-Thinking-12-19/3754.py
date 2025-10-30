class Solution:
    def maxDistance(self, s: str, k: int) -> int:
        n = len(s)

        # Vectors for movements (dx, dy)
        move_vec = {'N': (0, 1), 'S': (0, -1), 'E': (1, 0), 'W': (-1, 0)}

        # Original prefix sums for x+y, x-y, -x+y, -x-y
        # P_orig[p][i] is the value for the p-th sum type at step i (after i movements)
        # Step 0 is at (0,0)
        P_orig = [[0] * (n + 1) for _ in range(4)]

        # Count of potential changes before step i that increase the sum type value by 2
        # S[p][i] is the count for the p-th sum type among s[0]..s[i-1]
        S = [[0] * (n + 1) for _ in range(4)]

        # Define which original moves decrease which sum type, thus providing +2 potential
        # Change from d1 to d2 results in delta = sum(vec(d2)) - sum(vec(d1))
        # To get max delta = 2, we must change a move with sum = -1 to a move with sum = 1.
        # Sum type 0 (x+y): u1 = dx+dy. u1=-1 for S, W. u1=1 for N, E.
        #   Decreasers (u1=-1): S, W. Potential change S/W -> N/E gives delta = 1 - (-1) = 2.
        # Sum type 1 (x-y): u2 = dx-dy. u2=-1 for N, W. u2=1 for S, E.
        #   Decreasers (u2=-1): N, W. Potential change N/W -> S/E gives delta = 1 - (-1) = 2.
        # Sum type 2 (-x+y): u3 = -dx+dy. u3=-1 for S, E. u3=1 for N, W.
        #   Decreasers (u3=-1): S, E. Potential change S/E -> N/W gives delta = 1 - (-1) = 2.
        # Sum type 3 (-x-y): u4 = -dx-dy. u4=-1 for N, E. u4=1 for S, W.
        #   Decreasers (u4=-1): N, E. Potential change N/E -> S/W gives delta = 1 - (-1) = 2.

        decreasers = [
            {'S', 'W'}, # x+y
            {'N', 'W'}, # x-y
            {'S', 'E'}, # -x+y
            {'N', 'E'}  # -x-y
        ]

        for i in range(n):
            dx, dy = move_vec[s[i]]

            # Update original prefix sums at step i+1
            P_orig[0][i+1] = P_orig[0][i] + (dx + dy) # x+y
            P_orig[1][i+1] = P_orig[1][i] + (dx - dy) # x-y
            P_orig[2][i+1] = P_orig[2][i] + (-dx + dy) # -x+y
            P_orig[3][i+1] = P_orig[3][i] + (-dx - dy) # -x-y

            # Update S counts at step i+1
            for p in range(4):
                # S[p][i+1] is count of decreasers among s[0]...s[i]
                S[p][i+1] = S[p][i] + (1 if s[i] in decreasers[p] else 0)

        # Calculate maximum achievable value for each of the four sum types (and their negatives)
        # using k changes.
        # The maximum possible value of max_i P'_p[i] using k changes is max_i (P_orig[p][i] + 2 * min(k, S[p][i]))
        # This is because to maximize P'_p[i] at a specific step i, we want to maximize the accumulated delta up to step i-1.
        # The max delta at step i-1 is achieved by applying changes that give +2 delta at indices j < i.
        # The number of such potential indices is S[p][i]. We can use at most k changes.
        # So max accumulated delta up to step i-1 is 2 * min(k, S[p][i]).

        max_vals = []
        for p in range(4):
            current_max = -float('inf')
            for i in range(n + 1):
                # Value at step i = original value + maximum possible bonus from k changes before step i
                val = P_orig[p][i] + 2 * min(k, S[p][i])
                current_max = max(current_max, val)
            max_vals.append(current_max)

        # The maximum Manhattan distance is max_i (|x_i| + |y_i|).
        # |x| + |y| = max(x+y, x-y, -x+y, -x-y).
        # max_i (|x_i|+|y_i|) = max_i max(P'_0[i], P'_1[i], P'_2[i], P'_3[i]).
        # The maximum possible value of max_i P'_p[i] is max_vals[p].
        # The maximum possible value of max_i (|P'_p[i]|) is max(max_i P'_p[i], max_i -P'_p[i]).
        # max_i -P'_0[i] = max_i P'_3[i]. Max possible value is max_vals[3].
        # max_i -P'_1[i] = max_i P'_2[i]. Max possible value is max_vals[2].

        # The maximum possible value for max_i |x_i+y_i| is max(max_i (x+y), max_i -(x+y)) = max(max_vals[0], max_vals[3]).
        # The maximum possible value for max_i |x_i-y_i| is max(max_i (x-y), max_i -(x-y)) = max(max_vals[1], max_vals[2]).
        
        # The overall maximum Manhattan distance is max(max_i |x_i+y_i| possible, max_i |x_i-y_i| possible).
        # This is max(max(max_vals[0], max_vals[3]), max(max_vals[1], max_vals[2])).
        # Which simplifies to max(max_vals[0], max_vals[1], max_vals[2], max_vals[3]).
        
        return max(max_vals)