from typing import List

class Solution:
  def countServers(self, n: int, logs: List[List[int]], x: int, queries: List[int]) -> List[int]:
    # Sort logs by time. logs[i] is [server_id, time].
    # We sort based on time, which is logs[i][1].
    logs.sort(key=lambda item: item[1])

    # Augment queries with their original indices to reconstruct the answer later.
    # Each element becomes a tuple: (query_time, original_index).
    indexed_queries = []
    for i in range(len(queries)):
        indexed_queries.append((queries[i], i))
    
    # Sort these augmented queries by query_time.
    # This is crucial for the sliding window approach, as it ensures T_end (query_time) 
    # is non-decreasing as we process queries.
    indexed_queries.sort(key=lambda item: item[0])

    # Initialize an array to store answers.
    # It has the same length as the original queries list.
    ans = [0] * len(queries)
    
    # server_freq[server_id] stores the count of logs for server_id currently in the window.
    # Server IDs are 1-indexed (from 1 to n), so the array size is n + 1.
    server_freq = [0] * (n + 1) 
    
    # active_servers_count stores the number of unique servers
    # that have at least one log message within the current time window.
    # A server is active if server_freq[server_id] > 0.
    active_servers_count = 0
    
    # Pointers for the sorted 'logs' array, to manage the sliding window.
    # left_ptr points to the current leftmost log in the logs array that might be part of the window.
    # right_ptr points to the log in the logs array that is next to be considered for adding to the window.
    left_ptr = 0
    right_ptr = 0 # This pointer will scan up to num_logs
    
    num_logs = len(logs)

    # Process each query (now sorted by their time T_end)
    for query_time, original_idx in indexed_queries:
        # Define the current query's time window [T_start, T_end].
        # The problem states intervals are inclusive: [queries[i] - x, queries[i]].
        T_start = query_time - x
        T_end = query_time
        
        # Phase 1: Add logs to the window.
        # These are logs whose time falls at or before T_end of the current query.
        # Iterate with right_ptr to include new logs into the window.
        while right_ptr < num_logs and logs[right_ptr][1] <= T_end:
            server_id = logs[right_ptr][0]
            if server_freq[server_id] == 0:
                # This server was not active, but a new log makes it active.
                active_servers_count += 1
            server_freq[server_id] += 1 # Increment count of this server's logs in window.
            right_ptr += 1 # Move to the next log.
        
        # Phase 2: Remove logs from the window.
        # These are logs whose time is now less than T_start of the current query.
        # Iterate with left_ptr to exclude old logs from the window.
        while left_ptr < num_logs and logs[left_ptr][1] < T_start:
            server_id = logs[left_ptr][0]
            server_freq[server_id] -= 1 # Decrement count of this server's logs.
            if server_freq[server_id] == 0:
                # This server has no more logs in the window; it becomes inactive.
                active_servers_count -= 1
            left_ptr += 1 # Move to the next log to consider removing.
        
        # The number of servers that did NOT receive any requests is
        # total servers (n) minus the count of servers that DID receive requests (active_servers_count).
        ans[original_idx] = n - active_servers_count
            
    return ans