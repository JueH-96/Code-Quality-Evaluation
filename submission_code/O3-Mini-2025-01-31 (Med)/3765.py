from typing import List
import math

# We'll implement a Convex Hull Trick structure which supports adding lines
# with monotonic slopes and querying the minimum value at a given x.
class ConvexHullTrick:
    # Each line is represented as (m, b) for function f(x)= m*x + b.
    def __init__(self):
        self.lines = []
        self.ptr = 0  # pointer for queries (x queries are non-decreasing)
        
    # Helper: compute intersection x-coordinate of two lines l1 and l2.
    def intersect(self, line1, line2):
        m1, b1 = line1
        m2, b2 = line2
        # m1*x + b1 = m2*x + b2  => x = (b2 - b1)/(m1 - m2)
        # (m1 > m2 normally for our use if slopes are in decreasing order)
        return (b2 - b1) / (m1 - m2)
    
    # Add a line (m, b) to the structure.
    # We assume that the slopes m are inserted in monotonic order (here, they will be in non-increasing order).
    def add_line(self, m, b):
        # If the new line never becomes useful (its intersection with the last line is
        # to the left of the last intersection), remove the last line.
        # We need to maintain the lower envelope.
        # For each new line, while there are at least one line in the hull,
        # check if its intersection with the last line is not to the right 
        # of the intersection between the last two lines.
        # Because our queries in x are in increasing order, we remove useless lines.
        new_line = (m, b)
        # While there are at least one line in the hull:
        while self.lines:
            # if new line and the last line are parallel, then keep the line with lower intercept.
            if abs(self.lines[-1][0] - m) < 1e-9:
                if b >= self.lines[-1][1]:
                    # new line is never better, so do not add.
                    return
                else:
                    self.lines.pop()
                    if self.ptr > len(self.lines) - 1:
                        self.ptr = len(self.lines) - 1
                    continue
            # If there is only one line, then compare directly.
            if len(self.lines) == 1:
                x = self.intersect(self.lines[-1], new_line)
                # if the new line's intersection is not to the right of the previous intersection, then pop.
                if x <= -1e18:
                    self.lines.pop()
                    self.ptr = 0
                    continue
                break
            # Otherwise, get the intersection of the last two lines.
            x_last = self.intersect(self.lines[-2], self.lines[-1])
            x_new = self.intersect(self.lines[-1], new_line)
            if x_new <= x_last + 1e-9:
                self.lines.pop()
                if self.ptr > len(self.lines) - 1:
                    self.ptr = len(self.lines) - 1
            else:
                break
        self.lines.append(new_line)
        
    # Query the minimum value at x.
    def query(self, x):
        # Since x queries are non-decreasing, we can move a pointer.
        if not self.lines:
            return math.inf
        # Move pointer as long as the next line is better at x.
        while self.ptr < len(self.lines)-1 and \
              self.lines[self.ptr+1][0]*x + self.lines[self.ptr+1][1] < self.lines[self.ptr][0]*x + self.lines[self.ptr][1]:
            self.ptr += 1
        return self.lines[self.ptr][0]*x + self.lines[self.ptr][1]

class Solution:
    def minimumCost(self, nums: List[int], cost: List[int], k: int) -> int:
        n = len(nums)
        # prefix sums of nums and cost; index 0 means sum over zero elements.
        P = [0]*(n+1)  # sum of nums[0..i-1]
        Q = [0]*(n+1)  # sum of cost[0..i-1]
        for i in range(1, n+1):
            P[i] = P[i-1] + nums[i-1]
            Q[i] = Q[i-1] + cost[i-1]
        
        # Let dp[t][i] be minimal cost dividing first i elements into exactly t segments.
        # We'll maintain dp[t] for t = 0,...,n. For i < t, dp[t][i] is not feasible.
        dp = [[math.inf]*(n+1) for _ in range(n+1)]
        dp[0][0] = 0  # 0 elements divided into 0 segments costs 0.
        
        # Compute dp[1][i] directly:
        # For t = 1, the only valid partition is to take [0, i-1] as one segment.
        # Its cost is: (P[i] + k * 1) * (Q[i] - Q[0]) = (P[i] + k)*Q[i]
        for i in range(1, n+1):
            dp[1][i] = (P[i] + k) * Q[i]
        
        # For t from 2 to n, compute dp[t][i] where i>=t.
        # dp[t][i] = (P[i] + k*t) * Q[i] + min_{j from t-1 to i-1}[ dp[t-1][j] - (P[i] + k*t)*Q[j] ]
        for t in range(2, n+1):
            # For a fixed t, we want to compute dp[t][i] for i from t to n.
            # We need to query for each i, the minimal value of: dp[t-1][j] - (P[i] + k*t)* Q[j], for j in [t-1, i-1].
            # We'll process i in increasing order and maintain a convex hull of lines corresponding to indices j.
            hull = ConvexHullTrick()
            # j will run from t-1 to n-1. Initially add j = t-1.
            # For each line corresponding to j, line function is: f(x) = dp[t-1][j] - x * Q[j]
            # where x will be (P[i] + k*t) for the query.
            j_index = t - 1
            # Add the line for j_index = t-1.
            hull.add_line(-Q[j_index], dp[t-1][j_index])
            # Now iterate i from t to n.
            for i in range(t, n+1):
                # We can add new lines for j = i-1 (since j must be < i)
                if i - 1 >= j_index + 1:
                    # Add lines for j_index from current up to i-1.
                    while j_index < i - 1:
                        j_index += 1
                        hull.add_line(-Q[j_index], dp[t-1][j_index])
                # Our query x is (P[i] + k*t)
                x_query = P[i] + k*t
                best = hull.query(x_query)
                dp[t][i] = (P[i] + k*t) * Q[i] + best
            # End of t loop
        
        # The answer is the minimum over all possible segment counts from 1 to n for dp[t][n].
        ans = min(dp[t][n] for t in range(1, n+1))
        return ans