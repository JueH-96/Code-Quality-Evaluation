from typing import List

class Solution:
    def maxScore(self, points: List[int], m: int) -> int:
        n = len(points)

        def check(x: int) -> bool:
            if x == 0:
                return True
            
            V = []
            for p in points:
                if p == 0:
                    if x > 0: return False # Impossible to get positive score with 0 points
                    V.append(0)
                else:
                    V.append((x + p - 1) // p)

            min_total_moves = float('inf')

            for e in range(n):
                # For a given endpoint e, delta_i = 1 if i < e else 0
                
                # Forward pass to find minimal L_i based on left-side constraints
                l_fwd = [0] * (n - 1)
                if n > 1:
                    # Constraint from V[0]: 1 + L_0 >= V[0] => L_0 >= V[0] - 1
                    l_fwd[0] = max(0, V[0] - 1)
                    # For i > 0, num_visits[i] = R_{i-1} + L_i >= V[i]
                    # R_{i-1} = L_{i-1} + delta_{i-1}
                    # L_{i-1} + delta_{i-1} + L_i >= V[i] => L_i >= V[i] - (L_{i-1} + delta_{i-1})
                    
                    r_prev = l_fwd[0] + (1 if 0 < e else 0)
                    for i in range(1, n - 1):
                        l_fwd[i] = max(0, V[i] - r_prev)
                        r_prev = l_fwd[i] + (1 if i < e else 0)

                # Backward pass to find minimal L_i based on right-side constraints
                l_bwd = [0] * (n - 1)
                if n > 1:
                    # Constraint from V[n-1]: R_{n-2} >= V[n-1]
                    # L_{n-2} + delta_{n-2} >= V[n-1]
                    delta_n_2 = 1 if n - 2 < e else 0
                    l_bwd[n-2] = max(0, V[n-1] - delta_n_2)
                    # For i < n-2, num_visits[i+1] = R_i + L_{i+1} >= V_{i+1}
                    # L_i + delta_i + L_{i+1} >= V_{i+1} => L_i >= V_{i+1} - (L_{i+1} + delta_i)
                    for i in range(n - 3, -1, -1):
                        delta_i_plus_1 = 1 if i + 1 < e else 0
                        l_bwd[i] = max(0, V[i+1] - (l_bwd[i+1] + delta_i_plus_1))
                
                # Combine passes and calculate total moves for this endpoint 'e'
                current_moves = 0
                if n > 1:
                    for i in range(n - 1):
                        l_i = max(l_fwd[i], l_bwd[i])
                        delta_i = 1 if i < e else 0
                        r_i = l_i + delta_i
                        current_moves += l_i + r_i
                
                min_total_moves = min(min_total_moves, current_moves)

            return min_total_moves <= m

        low = 0
        high = m * max(points) + max(points)
        ans = 0

        while low <= high:
            mid = (low + high) // 2
            if mid < 0: continue
            if check(mid):
                ans = mid
                low = mid + 1
            else:
                high = mid - 1
                
        return ans