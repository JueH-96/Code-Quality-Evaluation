from typing import List

class Solution:
    def minimumDistance(self, points: List[List[int]]) -> int:
        n = len(points)
        # For Manhattan distance, it is well‐known that:
        #    max_{i,j} (|x_i - x_j| + |y_i - y_j|) == max( max_i(x+y) - min_i(x+y),  max_i(x-y) - min_i(x-y) ).
        # Thus, we precompute two transforms per point.
        v1 = [p[0] + p[1] for p in points]
        v2 = [p[0] - p[1] for p in points]
        
        # We will compute for each of v1 and v2:
        #    max value, second max value, count for max;
        #    min value, second min value, count for min.
        INF = 10**20
        NEG_INF = -10**20
        
        # For v1
        max_v1 = NEG_INF
        second_max_v1 = NEG_INF
        count_max_v1 = 0
        min_v1 = INF
        second_min_v1 = INF
        count_min_v1 = 0
        
        # For v2
        max_v2 = NEG_INF
        second_max_v2 = NEG_INF
        count_max_v2 = 0
        min_v2 = INF
        second_min_v2 = INF
        count_min_v2 = 0
        
        # Process v1 values to compute extremes.
        for val in v1:
            # Update maximum for v1:
            if val > max_v1:
                second_max_v1 = max_v1
                max_v1 = val
                count_max_v1 = 1
            elif val == max_v1:
                count_max_v1 += 1
            elif val > second_max_v1:
                second_max_v1 = val
                
            # Update minimum for v1:
            if val < min_v1:
                second_min_v1 = min_v1
                min_v1 = val
                count_min_v1 = 1
            elif val == min_v1:
                count_min_v1 += 1
            elif val < second_min_v1:
                second_min_v1 = val
        
        # Process v2 values to compute extremes.
        for val in v2:
            # Update maximum for v2:
            if val > max_v2:
                second_max_v2 = max_v2
                max_v2 = val
                count_max_v2 = 1
            elif val == max_v2:
                count_max_v2 += 1
            elif val > second_max_v2:
                second_max_v2 = val
                
            # Update minimum for v2:
            if val < min_v2:
                second_min_v2 = min_v2
                min_v2 = val
                count_min_v2 = 1
            elif val == min_v2:
                count_min_v2 += 1
            elif val < second_min_v2:
                second_min_v2 = val
        
        # In cases where all points have identical values in a transform,
        # our second extreme might not have been updated; in that case, set it equal to extreme.
        if second_max_v1 == NEG_INF:
            second_max_v1 = max_v1
        if second_min_v1 == INF:
            second_min_v1 = min_v1
        if second_max_v2 == NEG_INF:
            second_max_v2 = max_v2
        if second_min_v2 == INF:
            second_min_v2 = min_v2
        
        # For every candidate point removal, we need to determine the new overall Manhattan "diameter"
        # (i.e. the maximum distance among the remaining points). Removing a point only affects the extremes
        # if that point is a unique extremum.
        best = INF
        for i in range(n):
            # For v1:
            if v1[i] == max_v1 and count_max_v1 == 1:
                new_max_v1 = second_max_v1
            else:
                new_max_v1 = max_v1
            if v1[i] == min_v1 and count_min_v1 == 1:
                new_min_v1 = second_min_v1
            else:
                new_min_v1 = min_v1
            diff1 = new_max_v1 - new_min_v1
            
            # For v2:
            if v2[i] == max_v2 and count_max_v2 == 1:
                new_max_v2 = second_max_v2
            else:
                new_max_v2 = max_v2
            if v2[i] == min_v2 and count_min_v2 == 1:
                new_min_v2 = second_min_v2
            else:
                new_min_v2 = min_v2
            diff2 = new_max_v2 - new_min_v2
            
            # The new overall diameter is the maximum of the two differences.
            candidate = diff1 if diff1 >= diff2 else diff2
            best = min(best, candidate)
        
        return best


# The code below is for local testing and reading from STDIN.
def solve():
    import sys
    data = sys.stdin.read().strip()
    if not data:
        return
    # The input is assumed to be a representation of the points list,
    # for example: [[3,10],[5,15],[10,2],[4,4]]
    points = eval(data)
    sol = Solution()
    res = sol.minimumDistance(points)
    sys.stdout.write(str(res))
    
if __name__ == '__main__':
    solve()