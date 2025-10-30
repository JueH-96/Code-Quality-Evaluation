class Solution:
    def maxDistance(self, s: str, k: int) -> int:
        n = len(s)

        # Precompute prefix sums for counts of each direction
        p_N = [0] * (n + 1)
        p_S = [0] * (n + 1)
        p_E = [0] * (n + 1)
        p_W = [0] * (n + 1)
        for i in range(n):
            p_N[i+1] = p_N[i] + (1 if s[i] == 'N' else 0)
            p_S[i+1] = p_S[i] + (1 if s[i] == 'S' else 0)
            p_E[i+1] = p_E[i] + (1 if s[i] == 'E' else 0)
            p_W[i+1] = p_W[i] + (1 if s[i] == 'W' else 0)
        
        # check(D) returns true if it's possible to achieve Manhattan distance D
        def check(D: int) -> bool:
            if D > n:
                return False

            # For a prefix of length m, check if distance D is achievable
            # We must have m >= D since max distance after m moves is m.
            for m in range(D, n + 1):
                # Count of moves that contribute positively to u and v in the original prefix
                c_pos_u = p_N[m] + p_E[m]
                c_pos_v = p_E[m] + p_S[m]
                
                # Effective target distance D_eff, ensuring same parity with m
                D_eff = D if (m - D) % 2 == 0 else D + 1
                if D_eff > m:
                    continue
                
                # Cost to make u >= D_eff
                # Required 'N' or 'E' moves: n_pos_u >= (m + D_eff) / 2
                # Cost is how many S/W we must change.
                cost1 = max(0, (m + D_eff) // 2 - c_pos_u)
                if cost1 <= k:
                    return True

                # Cost to make u <= -D_eff
                # Required 'N' or 'E' moves: n_pos_u <= (m - D_eff) / 2
                # Cost is how many N/E we must change.
                cost2 = max(0, c_pos_u - (m - D_eff) // 2)
                if cost2 <= k:
                    return True

                # Cost to make v >= D_eff
                # Required 'E' or 'S' moves: n_pos_v >= (m + D_eff) / 2
                cost3 = max(0, (m + D_eff) // 2 - c_pos_v)
                if cost3 <= k:
                    return True
                
                # Cost to make v <= -D_eff
                # Required 'E' or 'S' moves: n_pos_v <= (m - D_eff) / 2
                cost4 = max(0, c_pos_v - (m - D_eff) // 2)
                if cost4 <= k:
                    return True
            
            return False

        # Binary search for the maximum achievable distance
        low, high = 0, n
        ans = 0
        while low <= high:
            mid = (low + high) // 2
            if mid == 0:
                ans = max(ans, 0)
                low = mid + 1
                continue
            
            if check(mid):
                ans = mid
                low = mid + 1
            else:
                high = mid - 1
        
        return ans