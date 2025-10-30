from typing import List

class Solution:
    def countServers(self, n: int, logs: List[List[int]], x: int, queries: List[int]) -> List[int]:
        # Create a set to store the servers that have received requests
        servers_with_requests = set()
        
        # Create a dictionary to store the last request time for each server
        last_request_time = {}
        
        # Iterate over the logs to populate the set and dictionary
        for server_id, time in logs:
            servers_with_requests.add(server_id)
            last_request_time[server_id] = time
        
        # Initialize the result list
        result = []
        
        # Iterate over the queries
        for query in queries:
            # Initialize a set to store the servers that have received requests in the current time interval
            servers_with_requests_in_interval = set()
            
            # Iterate over the logs to populate the set
            for server_id, time in logs:
                if query - x <= time <= query:
                    servers_with_requests_in_interval.add(server_id)
            
            # Calculate the number of servers that did not receive any requests in the current time interval
            num_servers_without_requests = n - len(servers_with_requests_in_interval)
            
            # Append the result to the list
            result.append(num_servers_without_requests)
        
        # Return the result list
        return result