import heapq

class Solution:
  def earliestSecondToMarkIndices(self, nums: List[int], changeIndices: List[int]) -> int:
    n = len(nums)
    m = len(changeIndices)

    def check(k_seconds: int) -> bool:
      # k_seconds is the number of 1-indexed seconds allowed (1 to k_seconds).
      # This corresponds to 0-indexed time steps 0 to k_seconds-1.
      
      # Step 1: Determine Deadlines
      # last_potential_mark_time[i] stores the latest time s (0-indexed)
      # in [0, k_seconds-1] where index i can be marked.
      last_potential_mark_time = [-1] * n 
      for s_idx in range(k_seconds):
          # changeIndices values are 1-indexed, convert to 0-indexed target for nums
          target_idx = changeIndices[s_idx] - 1 
          last_potential_mark_time[target_idx] = s_idx
      
      # Step 2: Group Tasks by Deadline
      # events_at_time[s] will store a list of costs (nums[i]) for indices i
      # whose last_potential_mark_time[i] == s.
      events_at_time = [[] for _ in range(k_seconds)]
      for i in range(n):
          if last_potential_mark_time[i] == -1:
              # Index i does not appear in changeIndices[0...k_seconds-1]
              return False 
          
          deadline = last_potential_mark_time[i]
          # The cost is nums[i] decrements. If nums[i] is 0, cost is 0.
          events_at_time[deadline].append(nums[i])

      # Step 3: Greedy Scheduling
      active_decrement_tasks_pq = []  # Min-priority queue for costs nums[i]
      decrements_available = 0      # Free seconds available for decrements
      
      # Iterate through time s from 0 to k_seconds-1
      for s in range(k_seconds):
          # Add tasks whose deadline is today (time s) to the priority queue
          for cost in events_at_time[s]:
              heapq.heappush(active_decrement_tasks_pq, cost)
          
          if not active_decrement_tasks_pq:
              # No tasks are pending. This second 's' can be a free slot.
              decrements_available += 1
          else:
              # A task must be marked. Pick the one with the smallest cost from PQ.
              cost_to_pay = heapq.heappop(active_decrement_tasks_pq)
              
              if decrements_available < cost_to_pay:
                  # Not enough free seconds accumulated to pay for its decrements.
                  return False
              
              decrements_available -= cost_to_pay
              # Second 's' is used for this mark operation, so it doesn't add to decrements_available.
      
      # Step 4: If loop completes, all marks scheduled successfully.
      return True

    # Binary search for the earliest k_seconds
    ans = -1
    low = 1  # Minimum possible seconds is 1 (if n=1, nums[0]=0, etc.)
    high = m # Maximum possible seconds is m

    while low <= high:
      mid_k_seconds = low + (high - low) // 2
      if mid_k_seconds == 0: # Should not happen if low starts at 1
          # If it could, and n > 0, it's impossible. If n = 0, it's possible (0 seconds).
          # Constraints: 1 <= n, m. So mid_k_seconds >= 1.
          is_possible = False
      else:
          is_possible = check(mid_k_seconds)
      
      if is_possible:
        ans = mid_k_seconds
        high = mid_k_seconds - 1  # Try to find an even earlier second
      else:
        low = mid_k_seconds + 1   # Need more seconds
        
    return ans