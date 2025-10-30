class Solution:
    def minNumberOfSeconds(self, mountainHeight: int, workerTimes: List[int]) -> int:
        import heapq
        
        # Create a max-heap (negative values because Python has a min-heap by default)
        max_heap = []
        for time in workerTimes:
            heapq.heappush(max_heap, (-time, time))  # (current negative time cost, base time)
        
        # Total seconds taken by each worker
        worker_seconds = [0] * len(workerTimes)
        
        # Distribute the work of reducing the mountain height
        for _ in range(mountainHeight):
            # Get the worker with the minimum current time cost (max heap so we use negative)
            current_time_cost, base_time = heapq.heappop(max_heap)
            current_time_cost = -current_time_cost  # Convert back to positive
            
            # Assign one unit of work to this worker
            worker_seconds.append(current_time_cost)
            
            # Increase the worker's next time cost
            new_time_cost = current_time_cost + base_time
            
            # Push the updated time cost back to the heap
            heapq.heappush(max_heap, (-new_time_cost, base_time))
        
        # The result is the maximum time any worker has taken
        return max(worker_seconds)