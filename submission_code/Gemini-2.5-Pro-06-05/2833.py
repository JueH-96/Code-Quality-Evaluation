from typing import List

class Solution:
  def countServers(self, n: int, logs: List[List[int]], x: int, queries: List[int]) -> List[int]:
    """
    Calculates the number of idle servers for each query's time window using a 
    sliding window approach.
    """
    
    # Sort logs by time to enable efficient chronological processing.
    logs.sort(key=lambda item: item[1])

    # Pair each query with its original index and sort by query time.
    # This allows processing queries in increasing order of time while preserving
    # the ability to return results in the original order.
    # enumerate(queries) gives (index, value) pairs. We sort by the value (p[1]).
    indexed_queries = sorted(enumerate(queries), key=lambda p: p[1])
    
    ans = [0] * len(queries)
    server_counts = {}
    
    # Pointers for the sliding window on the `logs` array.
    left = 0
    right = 0
    
    # Process queries in their sorted order.
    for original_index, q_time in indexed_queries:
        start_time = q_time - x
        end_time = q_time
        
        # --- Slide the window ---
        
        # 1. Expand window from the right: Add logs that are now within the window.
        while right < len(logs) and logs[right][1] <= end_time:
            server_id = logs[right][0]
            server_counts[server_id] = server_counts.get(server_id, 0) + 1
            right += 1
        
        # 2. Shrink window from the left: Remove logs that are no longer in the window.
        while left < right and logs[left][1] < start_time:
            server_id = logs[left][0]
            server_counts[server_id] -= 1
            if server_counts[server_id] == 0:
                del server_counts[server_id]
            left += 1
        
        # --- Calculate the result for the current query ---
        
        # The number of unique servers with requests in the current window.
        active_servers = len(server_counts)
        
        # The number of idle servers is the total minus the active ones.
        # Store the result at the query's original index.
        ans[original_index] = n - active_servers
        
    return ans