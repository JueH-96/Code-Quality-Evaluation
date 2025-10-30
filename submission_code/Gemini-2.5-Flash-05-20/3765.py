import collections

class Solution:
    def minimumCost(self, nums: List[int], cost: List[int], k: int) -> int:
        n = len(nums)

        # 1. Precompute prefix sums
        prefix_nums = [0] * (n + 1)
        prefix_cost = [0] * (n + 1)
        for i in range(n):
            prefix_nums[i+1] = prefix_nums[i] + nums[i]
            prefix_cost[i+1] = prefix_cost[i] + cost[i]

        # dp_prev[i] stores min cost to partition nums[0...i-1] into j-1 subarrays
        dp_prev = [float('inf')] * (n + 1)
        dp_prev[0] = 0 # Cost to partition empty prefix into 0 subarrays is 0

        min_total_cost = float('inf')

        # Iterate j from 1 to n (j represents the number of subarrays)
        for j in range(1, n + 1):
            # dp_curr[i] stores min cost to partition nums[0...i-1] into j subarrays
            dp_curr = [float('inf')] * (n + 1)
            
            # Deque for Convex Hull Trick optimization
            # Stores indices p representing lines L_p(x_query) = (-prefix_cost[p]) * x_query + dp_prev[p]
            # Slopes (-prefix_cost[p]) are non-increasing.
            # Queries (prefix_nums[i] + k * j) are non-decreasing.
            # We maintain a lower envelope (upper hull of points (prefix_cost[p], dp_prev[p])).
            deque = collections.deque()

            # Iterate i from j to n (i represents the end index + 1 of the prefix nums[0...i-1])
            # The current subarray ends at index i-1.
            # The previous prefix ends at index p_add-1 (i.e., state dp_prev[p_add]).
            for i in range(j, n + 1):
                # Add line to deque for dp_prev[p_add]
                # p_add represents the splitting point for the current subarray (nums[p_add ... i-1])
                # so p_add is the index where the previous j-1 subarrays end.
                # Valid p_add ranges from j-1 up to i-1.
                # Here, we add the line for the immediately preceding index, i.e., i-1.
                p_add = i - 1
                if dp_prev[p_add] != float('inf'):
                    # The line to add is L_p_add(x) = (-prefix_cost[p_add]) * x + dp_prev[p_add]
                    # We maintain the upper envelope of points (prefix_cost[p], dp_prev[p]) for minimization with increasing query slope.
                    # Or, more directly, remove p2 if (p1, p2, p_add) forms a non-convex shape (p2 is redundant).
                    # Condition: (Y_p2 - Y_p1) / (X_p2 - X_p1) >= (Y_p_add - Y_p2) / (X_p_add - X_p2)
                    # Using cross-product to avoid division (X values are prefix_cost, Y values are dp_prev):
                    # (dp_prev[p2] - dp_prev[p1]) * (prefix_cost[p_add] - prefix_cost[p2]) >= \
                    # (dp_prev[p_add] - dp_prev[p2]) * (prefix_cost[p2] - prefix_cost[p1])
                    while len(deque) >= 2:
                        p1 = deque[-2]
                        p2 = deque[-1]
                        if (dp_prev[p2] - dp_prev[p1]) * (prefix_cost[p_add] - prefix_cost[p2]) >= \
                           (dp_prev[p_add] - dp_prev[p2]) * (prefix_cost[p2] - prefix_cost[p1]):
                            deque.pop()
                        else:
                            break
                    deque.append(p_add)

                # Query phase: find the best previous state dp_prev[best_p]
                query_slope = prefix_nums[i] + k * j
                
                if not deque:
                    # If deque is empty, no valid previous states for j-1 subarrays found
                    # dp_curr[i] remains infinity.
                    continue

                # Remove lines from the front of the deque that are no longer optimal
                # for the current increasing query_slope.
                # If L_p1(query_slope) >= L_p2(query_slope), p1 is worse or equal, so pop p1.
                # (-prefix_cost[p1] * query_slope + dp_prev[p1]) >= (-prefix_cost[p2] * query_slope + dp_prev[p2])
                while len(deque) >= 2:
                    p1 = deque[0]
                    p2 = deque[1]
                    if (dp_prev[p1] - prefix_cost[p1] * query_slope) >= \
                       (dp_prev[p2] - prefix_cost[p2] * query_slope):
                        deque.popleft()
                    else:
                        break
                
                best_p = deque[0]
                
                # If the best previous state is still infinity, it's not a valid path
                if dp_prev[best_p] == float('inf'):
                    continue

                # Calculate dp_curr[i] using the optimal best_p
                calculated_term = dp_prev[best_p] - prefix_cost[best_p] * query_slope
                dp_curr[i] = (prefix_nums[i] + k * j) * prefix_cost[i] + calculated_term

            # After filling dp_curr for the current j, update dp_prev for the next iteration (j+1)
            dp_prev = dp_curr
            
            # Update overall minimum cost if dp_curr[n] is a valid solution
            if dp_curr[n] != float('inf'):
                min_total_cost = min(min_total_cost, dp_curr[n])
        
        return min_total_cost