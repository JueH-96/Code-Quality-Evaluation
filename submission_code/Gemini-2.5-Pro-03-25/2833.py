import collections
from typing import List

class Solution:
    """
    Solves the problem of counting inactive servers within given time windows.
    Uses a sliding window approach over sorted logs, coordinated with sorted queries.
    The time complexity is dominated by sorting, O(M log M + Q log Q), where M is the number of logs
    and Q is the number of queries. The space complexity is O(Q + N) for storing sorted queries, 
    results, and the server counts map (in the worst case, all N servers could be active).
    """
    def countServers(self, n: int, logs: List[List[int]], x: int, queries: List[int]) -> List[int]:
        """
        Calculates the number of servers that did not receive requests in specific time intervals.

        Args:
            n: The total number of servers. Server IDs are assumed to be 1 to n.
            logs: A list of [server_id, time] entries representing server requests. 
                  The list is not guaranteed to be sorted.
            x: The duration of the time window relative to the query time. The interval is 
               [query_time - x, query_time].
            queries: A list of query times.

        Returns:
            A list where each element corresponds to a query in the original `queries` list, 
            indicating the count of servers that received zero requests during the interval 
            [queries[i] - x, queries[i]].
        """

        # Sort logs by time. This allows us to efficiently process logs using a sliding window
        # as query times increase. Sorting is done in-place.
        # If modifying the input `logs` is not allowed, make a copy first: `sorted_logs = sorted(logs, key=lambda item: item[1])`
        logs.sort(key=lambda item: item[1])

        # Create pairs of (query_time, original_index) to process queries in increasing order of time,
        # while being able to place the results back into the correct position corresponding to the original query order.
        indexed_queries = []
        for i in range(len(queries)):
            indexed_queries.append((queries[i], i)) # Store tuple (query_time, original_index)
        
        # Sort the queries based on their time.
        indexed_queries.sort(key=lambda item: item[0])

        # Initialize the result array with zeros. This array will store the answer for each query.
        result = [0] * len(queries) 
        
        # Use collections.defaultdict(int) to store the count of logs for each server currently within the sliding window.
        # A server ID maps to the number of its logs whose timestamps fall within the current window.
        server_counts = collections.defaultdict(int) 
        
        # Keep track of the number of distinct servers that have at least one log entry 
        # currently within the active time window.
        active_server_count = 0 
        
        # Initialize two pointers for the sliding window over the sorted `logs` array.
        # `log_ptr_left` points to the first log entry that might still be in the window. Logs before this index are definitely out.
        # `log_ptr_right` points to the next log entry to consider adding to the window. Logs from `log_ptr_left` to `log_ptr_right - 1` are potentially in the window.
        log_ptr_left = 0  
        log_ptr_right = 0 

        # Process the queries in their sorted order (by time).
        for query_time, original_index in indexed_queries:
            # Define the time window for the current query. The interval is inclusive: [start_time, end_time].
            start_time = query_time - x
            end_time = query_time 

            # --- Phase 1: Expand the window (Move the right pointer) ---
            # Add logs to the window if their timestamp is less than or equal to the current query's end_time.
            # The `log_ptr_right` pointer advances through the sorted logs.
            while log_ptr_right < len(logs) and logs[log_ptr_right][1] <= end_time:
                server_id = logs[log_ptr_right][0]
                
                # If this server's count was 0 before adding this log, it means the server is becoming active
                # within the window (it wasn't active before, or its previous logs aged out).
                if server_counts[server_id] == 0:
                    active_server_count += 1
                    
                # Increment the count of logs for this server within the window.
                server_counts[server_id] += 1 
                
                # Advance the right pointer to consider the next log entry.
                log_ptr_right += 1 

            # --- Phase 2: Shrink the window (Move the left pointer) ---
            # Remove logs from the window if their timestamp is strictly less than the current query's start_time.
            # This means they have aged out of the sliding window. The `log_ptr_left` pointer advances.
            while log_ptr_left < log_ptr_right and logs[log_ptr_left][1] < start_time:
                server_id = logs[log_ptr_left][0]

                # Decrement the count of logs for this server as this specific log entry is leaving the window.
                server_counts[server_id] -= 1 
                
                # If this server's count drops to 0, it means it no longer has any logs within the current window,
                # so it becomes inactive.
                if server_counts[server_id] == 0:
                    active_server_count -= 1
                    
                # Advance the left pointer, effectively removing this log from the window's consideration.
                log_ptr_left += 1 
                
            # --- Phase 3: Calculate and store the result ---
            # The number of inactive servers is the total number of servers (`n`) minus the count of servers
            # currently active within the window (`active_server_count`).
            inactive_count = n - active_server_count
            
            # Store this result in the `result` array at the position corresponding to the original query's index.
            result[original_index] = inactive_count

        # Return the final list containing results for all queries, ordered according to the original `queries` list.
        return result