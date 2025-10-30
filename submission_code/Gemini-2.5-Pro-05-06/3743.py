import math

class Solution:
  def maxFreeTime(self, eventTime: int, k: int, startTime: List[int], endTime: List[int]) -> int:
    n = len(startTime)
    if n == 0:
      return eventTime

    durations = [endTime[i] - startTime[i] for i in range(n)]

    # LS[idx][j] = min end time of meeting idx, using j reschedules for meetings 0...idx
    # all_LS_values[idx][j] stores this.
    # To handle LS[-1][j] (base case for meeting 0):
    # LS_prev_row[j] will represent LS[idx-1][j] in the loop.
    # Initialize LS_prev_row for the state before meeting 0: end time is 0, 0 moves.
    LS_prev_row = {0: 0} 
    for j_moves in range(1, k + 1):
        LS_prev_row[j_moves] = float('inf') # Other counts of moves are infinitely costly

    all_LS_values = [[float('inf')] * (k + 1) for _ in range(n)]

    for idx in range(n): # Current meeting index
      LS_curr_row = {} # Stores LS[idx][j]
      
      # Option 1: meeting idx is NOT rescheduled
      # Number of reschedules j remains same as for LS[idx-1][j].
      for j_moves in range(k + 1):
        prev_end_time = LS_prev_row.get(j_moves, float('inf'))
        if prev_end_time <= startTime[idx]: # Previous meeting must end before current one starts
          current_val = LS_curr_row.get(j_moves, float('inf'))
          LS_curr_row[j_moves] = min(current_val, endTime[idx])
      
      # Option 2: meeting idx IS rescheduled
      # Number of reschedules j for LS[idx][j] means j-1 for LS[idx-1][j-1].
      for j_moves in range(1, k + 1): # Need at least 1 move for current meeting
        prev_end_time_if_packed = LS_prev_row.get(j_moves - 1, float('inf'))
        if prev_end_time_if_packed != float('inf'):
          # Meeting idx is packed right after previous one
          current_meeting_end_time = prev_end_time_if_packed + durations[idx]
          if current_meeting_end_time <= eventTime: # Must be within event bounds
            current_val = LS_curr_row.get(j_moves, float('inf'))
            LS_curr_row[j_moves] = min(current_val, current_meeting_end_time)
      
      for j_moves in range(k + 1):
          all_LS_values[idx][j_moves] = LS_curr_row.get(j_moves, float('inf'))
      LS_prev_row = LS_curr_row


    # RS[idx][j] = max start time of meeting idx, using j reschedules for meetings idx...n-1
    # Similar logic for RS, processing meetings from n-1 down to 0.
    # RS_next_row[j] will represent RS[idx+1][j].
    # Base case for state after meeting n-1: start time is eventTime, 0 moves.
    RS_next_row = {0: eventTime}
    for j_moves in range(1, k + 1):
        RS_next_row[j_moves] = float('-inf')

    all_RS_values = [[float('-inf')] * (k + 1) for _ in range(n)]

    for idx in range(n - 1, -1, -1): # Current meeting index
      RS_curr_row = {} # Stores RS[idx][j]
      
      # Option 1: meeting idx is NOT rescheduled
      for j_moves in range(k + 1):
          next_meeting_start_time = RS_next_row.get(j_moves, float('-inf'))
          if next_meeting_start_time >= endTime[idx]: # Next meeting must start after current one ends
              current_val = RS_curr_row.get(j_moves, float('-inf'))
              RS_curr_row[j_moves] = max(current_val, startTime[idx])

      # Option 2: meeting idx IS rescheduled
      for j_moves in range(1, k + 1):
          next_meeting_start_time_if_packed = RS_next_row.get(j_moves - 1, float('-inf'))
          if next_meeting_start_time_if_packed != float('-inf'):
              # Meeting idx is packed right before next one
              current_meeting_start_time = next_meeting_start_time_if_packed - durations[idx]
              if current_meeting_start_time >= 0: # Must be within event bounds
                  current_val = RS_curr_row.get(j_moves, float('-inf'))
                  RS_curr_row[j_moves] = max(current_val, current_meeting_start_time)
      
      for j_moves in range(k + 1):
          all_RS_values[idx][j_moves] = RS_curr_row.get(j_moves, float('-inf'))
      RS_next_row = RS_curr_row

    max_free = 0

    # Case 1: Free time before the first meeting [0, s'[0]]
    # s'[0] is all_RS_values[0][j_moves]
    if n > 0: # only if there are meetings
        for j_moves in range(k + 1):
            s0 = all_RS_values[0][j_moves]
            if s0 != float('-inf'):
                max_free = max(max_free, s0 - 0)

    # Case 2: Free time after the last meeting [e'[n-1], eventTime]
    # e'[n-1] is all_LS_values[n-1][j_moves]
    if n > 0: # only if there are meetings
        for j_moves in range(k + 1):
            en_1 = all_LS_values[n-1][j_moves]
            if en_1 != float('inf'):
                max_free = max(max_free, eventTime - en_1)
        
    # Case 3: Free time between meeting idx and meeting idx+1
    # Interval is [e'[idx], s'[idx+1]]
    # e'[idx] is all_LS_values[idx][k_L]
    # s'[idx+1] is all_RS_values[idx+1][k_R]
    # Total reschedules k_L + k_R <= k
    for會議_idx in range(n - 1): # Gap is between meeting `會議_idx` and `會議_idx+1`
      for k_L in range(k + 1): # Moves for left part (meetings 0...會議_idx)
        e_prev = all_LS_values[會議_idx][k_L]
        if e_prev == float('inf'): # This configuration for left part is not possible
          continue
        
        # Budget for right part (meetings 會議_idx+1...n-1)
        k_R_budget = k - k_L 
        
        if k_R_budget < 0: # Not enough moves overall (k_L itself exceeds k)
            continue # This check is technically redundant if k_L loop is 0..k

        # We want to pick k_R for the right part such that 0 <= k_R <= k_R_budget
        # to maximize s_curr = all_RS_values[會議_idx+1][k_R].
        # Since all_RS_values[idx][j] is non-decreasing with j,
        # the maximum value for s_curr is achieved when k_R = k_R_budget.
        s_curr = all_RS_values[會議_idx+1][k_R_budget]
        
        if s_curr == float('-inf'): # This configuration for right part is not possible
          continue
        
        if s_curr >= e_prev: # Valid non-overlapping segment
          max_free = max(max_free, s_curr - e_prev)
              
    return max_free