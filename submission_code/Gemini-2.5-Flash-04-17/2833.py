from typing import List
import collections

class Solution:
    def countServers(self, n: int, logs: List[List[int]], x: int, queries: List[int]) -> List[int]:
        # 1. Sort logs by time
        # This allows us to use a sliding window approach based on time intervals.
        # Time complexity: O(L log L), where L is logs.length
        logs.sort(key=lambda log: log[1])

        # 2. Pair queries with their original indices and sort them by query time.
        # This is crucial for processing queries in a time-ordered manner
        # while preserving their original indices for the final output.
        # Time complexity: O(Q log Q), where Q is queries.length
        indexed_queries = sorted([(q, idx) for idx, q in enumerate(queries)])

        # 3. Initialize the results array.
        # This array will store the final answer for each query, placed at its
        # original index.
        # Space complexity: O(Q)
        results = [0] * len(queries)

        # 4. Initialize data structures for the sliding window.
        # server_counts: A dictionary to keep track of the number of log entries
        # for each server ID within the current time window [start_time, end_time].
        # Space complexity: O(min(N, L)) in the worst case, as it stores counts
        # for at most min(N, L) unique server IDs.
        server_counts = collections.defaultdict(int)
        
        # active_server_count: A counter for the number of unique server IDs
        # currently present in the server_counts dictionary (i.e., servers
        # that have at least one log entry in the current window).
        active_server_count = 0
        
        # left and right pointers define the boundaries [left, right) of the
        # sliding window on the sorted logs list.
        left = 0
        right = 0

        # 5. Iterate through the sorted queries. For each query, adjust the sliding
        # window to cover the relevant time interval [q - x, q].
        # The pointers 'left' and 'right' each traverse the 'logs' list at most once
        # across all queries. The operations inside the loops (dictionary access/update)
        # take O(1) on average.
        # Total time complexity for this loop: O(Q + L)
        for q, original_idx in indexed_queries:
            start_time = q - x
            end_time = q

            # --- Expand the window from the right ---
            # Add logs whose time is within the interval [start_time, end_time].
            # The right pointer moves forward as long as the log time is <= end_time.
            while right < len(logs) and logs[right][1] <= end_time:
                server_id = logs[right][0]
                
                # If this server ID is encountered for the first time in the window
                # (its count was 0 before adding this log), it becomes an active server.
                if server_counts[server_id] == 0:
                    active_server_count += 1
                    
                # Increment the count for this server ID in the window.
                server_counts[server_id] += 1
                
                # Move the right pointer to consider the next log.
                right += 1

            # --- Shrink the window from the left ---
            # Remove logs whose time is no longer within the interval [start_time, end_time).
            # The left pointer moves forward as long as the log time is < start_time.
            # Logs with time exactly equal to start_time should remain in the window.
            while left < len(logs) and logs[left][1] < start_time:
                server_id = logs[left][0]
                
                # Decrement the count for this server ID as it's leaving the window.
                server_counts[server_id] -= 1
                
                # If the count for this server ID becomes 0 after decrementing,
                # it means there are no more logs for this server in the current window.
                # This server is no longer active in the window.
                if server_counts[server_id] == 0:
                    active_server_count -= 1
                    # We could optionally remove the key if server_counts[server_id] == 0,
                    # but defaultdict handles count == 0 correctly for future access.
                    
                # Move the left pointer to exclude this log from the window.
                left += 1

            # After adjusting the window [left, right), 'active_server_count' holds the number
            # of unique servers that had at least one log entry with a timestamp 't'
            # such that q - x <= t <= q.
            
            # The number of servers that did *not* receive any requests in this interval
            # is the total number of servers 'n' minus the number of unique servers
            # that *did* receive requests.
            results[original_idx] = n - active_server_count

        # 6. Return the final results array. It is now populated according to
        # the original order of the queries input.
        return results