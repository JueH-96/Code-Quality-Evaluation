class Solution:
    def maxDistance(self, s: str, k: int) -> int:
        # We track the current position (X, Y) and for each of the 4 sign‐combinations
        # A = (Ax, Ay) in {(1,1), (1,-1), (-1,1), (-1,-1)},
        # we count how many prefix moves can be changed to improve A·(X,Y).
        # It turns out each original move v yields a possible "gain" of 2
        # in direction A exactly when v·A = -1, and 0 otherwise.
        # Let cnt2[id] = number of prefix moves with gain=2 for combination id.
        # Then the best total gain we can get (using up to k changes) is 2*min(cnt2[id], k).
        # Finally the best L1‐distance at any prefix t for combination A is
        #    A_x * X_t + A_y * Y_t  +  2 * min(cnt2[id], k).
        # We maintain X_t, Y_t and cnt2[id] on the fly and take a running maximum.
        
        # Pre‐map for each char which of the 4 combinations yield gain=2
        # id: 0→(1,1), 1→(1,-1), 2→(-1,1), 3→(-1,-1)
        gain_map = {
            'N': [1, 3],  # v = (0,1), v·A = A_y; v·A = -1 iff A_y = -1 => ids 1,3
            'S': [0, 2],  # v = (0,-1), v·A = -A_y; = -1 iff A_y = 1 => ids 0,2
            'E': [2, 3],  # v = (1,0), v·A = A_x; = -1 iff A_x = -1 => ids 2,3
            'W': [0, 1],  # v = (-1,0), v·A = -A_x; = -1 iff A_x = 1 => ids 0,1
        }
        # The 4 sign combinations:
        A = [(1, 1), (1, -1), (-1, 1), (-1, -1)]
        
        cnt2 = [0, 0, 0, 0]  # how many gains=2 so far in each of the 4 combos
        X = 0
        Y = 0
        ans = 0
        
        for ch in s:
            # update position
            if ch == 'N':
                Y += 1
            elif ch == 'S':
                Y -= 1
            elif ch == 'E':
                X += 1
            else:  # 'W'
                X -= 1
            
            # update counts of gain‐2 for each combination
            for idx in gain_map[ch]:
                cnt2[idx] += 1
            
            # compute the best possible L1 distance at this prefix
            for idx, (ax, ay) in enumerate(A):
                base = ax * X + ay * Y
                # we can realize at most 2 * min(cnt2[idx], k) extra gain
                gain = 2 * min(cnt2[idx], k)
                ans = max(ans, base + gain)
        
        return ans