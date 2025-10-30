import collections

class Solution:
  def minimumCost(self, nums: list[int], cost: list[int], k: int) -> int:
    N = len(nums)

    ps_nums = [0] * (N + 1)
    ps_cost = [0] * (N + 1)
    for i in range(N):
      ps_nums[i+1] = ps_nums[i] + nums[i]
      ps_cost[i+1] = ps_cost[i] + cost[i]

    # dp[i][p] = min cost for nums[0...i-1] (length i) using p subarrays
    dp = [[float('inf')] * (N + 1) for _ in range(N + 1)]
    dp[0][0] = 0 # Base case: 0 elements, 0 subarrays, 0 cost.

    Line = collections.namedtuple('Line', ['m', 'b'])

    def eval_line(line: Line, x: int) -> int:
      return line.m * x + line.b

    def add_line_to_dq(dq: collections.deque, new_line: Line):
      # Slopes ps_cost[j] are strictly increasing for j >= 1. ps_cost[0]=0.
      # So, new_line.m will be > dq[-1].m unless dq is empty or dq[-1].m is also ps_cost[0].
      # This implies no two lines will have same slope m > 0.
      # Only case for same slope is m=0, if dp[0][p-1] is finite for multiple p, but it's only for dp[0][0].
      
      while len(dq) >= 2:
        l1, l2 = dq[-2], dq[-1]
        # Condition to remove l2: new_line makes l2 non-optimal.
        # Intersection_X(L_a, L_b) = (b_b - b_a) / (m_a - m_b)
        # Remove l2 if Intersection_X(l1, l2) >= Intersection_X(l2, new_line)
        # (l2.b - l1.b) / (l1.m - l2.m) >= (new_line.b - l2.b) / (l2.m - new_line.m)
        # Since l1.m < l2.m < new_line.m:
        #   (l1.m - l2.m) < 0
        #   (l2.m - new_line.m) < 0
        #   Product (l1.m - l2.m)*(l2.m - new_line.m) is positive.
        # Multiply by this positive product:
        # (l2.b - l1.b) * (l2.m - new_line.m) >= (new_line.b - l2.b) * (l1.m - l2.m)
        if (l2.b - l1.b) * (l2.m - new_line.m) >= \
           (new_line.b - l2.b) * (l1.m - l2.m):
          dq.pop()
        else:
          break
      dq.append(new_line)

    def query_dq(dq: collections.deque, x: int) -> int:
      if not dq:
        return float('inf')
      
      # Query points x are decreasing.
      while len(dq) >= 2:
        l1, l2 = dq[0], dq[1]
        # If l1 is not better than l2 at current x, remove l1.
        if eval_line(l1, x) >= eval_line(l2, x): 
          dq.popleft()
        else:
          break
      return eval_line(dq[0], x)

    for p_count in range(1, N + 1): 
      if p_count == 1:
        for i in range(1, N + 1):
          dp[i][p_count] = (ps_nums[i] + k) * ps_cost[i]
        continue 

      dq = collections.deque()
      # j_prev_end_idx_ptr tracks the next j for which line related to dp[j][p_count-1] is to be added.
      # Smallest j for dp[j][p_count-1] is j = p_count-1.
      j_prev_end_idx_ptr = p_count - 1 
      
      for i in range(p_count, N + 1): 
        # Add lines for dp[k_idx][p_count-1] where k_idx is from current j_prev_end_idx_ptr up to i-1.
        # temp_k_idx loops through these k_idx values.
        temp_k_idx_for_line_add = j_prev_end_idx_ptr
        while temp_k_idx_for_line_add < i:
          if dp[temp_k_idx_for_line_add][p_count-1] != float('inf'):
            line_to_add = Line(ps_cost[temp_k_idx_for_line_add], dp[temp_k_idx_for_line_add][p_count-1])
            add_line_to_dq(dq, line_to_add)
          temp_k_idx_for_line_add += 1
        j_prev_end_idx_ptr = temp_k_idx_for_line_add 
        
        if not dq: 
          dp[i][p_count] = float('inf')
          continue

        C_ip = ps_nums[i] + k * p_count 
        x_query = -C_ip
        
        min_term_val = query_dq(dq, x_query)
        
        if min_term_val == float('inf'): # Should only happen if dq was empty.
          dp[i][p_count] = float('inf')
          continue
        
        dp[i][p_count] = C_ip * ps_cost[i] + min_term_val

    min_total_cost = float('inf')
    for p_val in range(1, N + 1):
      min_total_cost = min(min_total_cost, dp[N][p_val])
      
    return int(min_total_cost)