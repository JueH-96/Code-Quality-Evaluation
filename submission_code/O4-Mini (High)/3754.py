class Solution:
    def maxDistance(self, s: str, k: int) -> int:
        # We use the fact that |x|+|y| = max over (a,b) in {(±1,±1)} of (a*x + b*y).
        # For each of the 4 choices of (a,b), track:
        #   f0 = sum_{i< t} (a*dx_i + b*dy_i), where (dx_i,dy_i) is the original step,
        #   bad = count of steps with projection = -1 (those that decrease a*x+b*y).
        # You can change up to k of those bad steps into good ones, each giving +2 in projection.
        # So the best projection at time t is f0 + 2*min(k, bad).
        # Take the maximum over all t and over all 4 directions.
        
        # Coefficients for the 4 directions:
        a = [1,  1, -1, -1]
        b = [1, -1,  1, -1]
        # bad_count[d] = number of steps so far whose projection onto (a[d],b[d]) is -1
        bad_count = [0,0,0,0]
        # f0[d] = sum of projections a[d]*dx + b[d]*dy so far
        f0 = [0,0,0,0]
        
        ans = 0
        for c in s:
            # original step vector
            if c == 'N':
                ox, oy = 0, 1
            elif c == 'S':
                ox, oy = 0, -1
            elif c == 'E':
                ox, oy = 1, 0
            else:  # 'W'
                ox, oy = -1, 0
            
            # update each of the 4 directional projections
            for d in range(4):
                proj = a[d]*ox + b[d]*oy  # this is ±1
                f0[d] += proj
                if proj == -1:
                    bad_count[d] += 1
                # we can flip up to k of the 'bad' steps to 'good', each flip adds +2
                flips = bad_count[d] if bad_count[d] < k else k
                ft = f0[d] + 2 * flips
                if ft > ans:
                    ans = ft
        
        return ans