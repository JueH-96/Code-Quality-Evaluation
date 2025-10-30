import collections
from typing import List

class Solution:
    def minimumCost(self, nums: List[int], cost: List[int], k: int) -> int:
        n = len(nums)
        
        prefix_N = [0] * (n + 1)
        prefix_C = [0] * (n + 1)
        for i in range(n):
            prefix_N[i+1] = prefix_N[i] + nums[i]
            prefix_C[i+1] = prefix_C[i] + cost[i]
            
        # dp_prev stores dp[p-1][...], dp_curr stores dp[p][...]
        dp_prev = [float('inf')] * (n + 1)
        dp_prev[0] = 0
        
        min_costs_at_n = []

        for p in range(1, n + 1):
            dp_curr = [float('inf')] * (n + 1)
            hull = collections.deque() # Stores lines (m, c) for the convex hull

            # Online CHT: Add line for j=i-1, then query for i.
            for i in range(p, n + 1):
                # Add the line corresponding to the previous split point j = i-1
                j = i - 1
                if dp_prev[j] != float('inf'):
                    m_new = -prefix_C[j]
                    c_new = dp_prev[j] - k * p * prefix_C[j]
                    
                    # Maintain the convexity of the lower envelope.
                    # Pop lines from the back that are made redundant by the new line.
                    while len(hull) >= 2:
                        m1, c1 = hull[-2]
                        m2, c2 = hull[-1]
                        # Compare intersection points without division
                        if (c2 - c1) * (m2 - m_new) >= (c_new - c2) * (m1 - m2):
                            hull.pop()
                        else:
                            break
                    hull.append((m_new, c_new))

                # Query the hull for the current end point i
                if not hull:
                    continue

                x_query = prefix_N[i]
                # Since query points are non-decreasing, pop lines from the front
                # that are no longer optimal.
                while len(hull) >= 2:
                    m1, c1 = hull[0]
                    m2, c2 = hull[1]
                    if m1 * x_query + c1 >= m2 * x_query + c2:
                        hull.popleft()
                    else:
                        break
                
                m_best, c_best = hull[0]
                min_val_from_j = m_best * x_query + c_best
                
                dp_curr[i] = (prefix_N[i] + k * p) * prefix_C[i] + min_val_from_j

            min_costs_at_n.append(dp_curr[n])
            dp_prev = dp_curr

        return min(min_costs_at_n)